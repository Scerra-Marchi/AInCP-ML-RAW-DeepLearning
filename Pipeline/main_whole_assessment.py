import argparse
import json
import os
import shutil
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from plotting import plot_corrcoeff, plot_dashboards
from test_classifier_regressor import test_classifier_regressor
from train_regressor import train_regressor
from train_select_classifiers import train_select_classifiers

RANDOM_STATE = 42
TOTAL_FOLDS = 10

# NOTE: As stated in README.md, update this path to your dataset location if needed.
DATA_FOLDER = "../Dataset/"

WINDOW_SIZE = 6400  # 4800 ≃ 180s, 6400 ≃ 240s, 8000 ≃ 300s
DECIMATION_FACTOR = 3

DEFAULT_ITERATIONS = TOTAL_FOLDS
CV_MIN_MEAN_TEST_SCORE = 0.7


def _subset_per_class(
    rng: np.random.Generator,
    indexes: np.ndarray,
    labels: np.ndarray,
    n_per_class: int,
) -> np.ndarray:
    out = []
    for label in np.unique(labels):
        group = indexes[labels == label]
        if group.size == 0:
            continue
        out.append(rng.choice(group, size=min(n_per_class, group.size), replace=False))
    return np.concatenate(out) if out else np.array([], dtype=int)


def _run_iteration(
    *,
    data_folder: str,
    save_folder: str,
    train_indexes: Iterable[int],
    test_indexes: Iterable[int],
    min_mean_test_score: float,
    window_size: int,
    decimation_factor: int,
    methods: list,
) -> None:
    os.makedirs(save_folder, exist_ok=True)

    best_estimators_csv = os.path.join(save_folder, "best_estimators_results.csv")
    if not os.path.exists(best_estimators_csv):
        print(" ----- TRAINING CLASSIFIERS ----- ")
        train_select_classifiers(
            data_folder,
            save_folder=save_folder,
            subjects_indexes=list(train_indexes),
            l_window_size=[window_size],
            l_method=methods,
            l_decimation_factor=[decimation_factor],
        )

    print(" ----- TRAINING REGRESSOR ----- ")
    train_regressor(
        data_folder,
        save_folder=save_folder,
        train_indexes=list(train_indexes),
        min_mean_test_score=min_mean_test_score,
        window_size=window_size,
        decimation_factor=decimation_factor,
    )

    combined_test_stats = os.path.join(save_folder, "combined_test_stats.json")
    if not os.path.exists(combined_test_stats):
        print(" ----- TESTING CLASSIFIER AND REGRESSOR ----- ")
        test_classifier_regressor(
            data_folder,
            save_folder=save_folder,
            test_indexes=list(test_indexes),
            min_mean_test_score=min_mean_test_score,
            window_size=window_size,
            decimation_factor=decimation_factor,
        )

    predictions_df = os.path.join(save_folder, "Week_stats", "predictions_dataframe.csv")
    if not os.path.exists(predictions_df):
        print(" ----- CREATING DASHBOARDS ----- ")
        plot_dashboards(
            data_folder,
            save_folder=save_folder,
            subjects_indexes=list(test_indexes),
            min_mean_test_score=min_mean_test_score,
            window_size=window_size,
            decimation_factor=decimation_factor,
        )


def _iteration_done(save_folder: str) -> bool:
    # The dashboard CSV is produced last; if it exists, the iteration is complete.
    predictions_df = os.path.join(save_folder, "Week_stats", "predictions_dataframe.csv")
    return os.path.exists(predictions_df)


def _load_or_create_iteration_split(
    *,
    iteration: int,
    save_folder: str,
    train_indexes: np.ndarray,
    test_indexes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    json_path = os.path.join(save_folder, "iteration_data.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as file:
            data = json.load(file)
        return np.array(data["Train Indexes"], dtype=int), np.array(data["Test Indexes"], dtype=int)

    data = {
        "Iteration": int(iteration),
        "Train Indexes": train_indexes.tolist(),
        "Test Indexes": test_indexes.tolist(),
    }
    with open(json_path, "w") as file:
        json.dump(data, file, indent=4)
    return train_indexes, test_indexes


def _aggregate_results(iterations_root: str, number_of_iterations: int) -> None:
    r2_list = []
    corrcoef_list = []
    for i in range(number_of_iterations):
        save_folder = os.path.join(iterations_root, f"Iteration_{i}")
        json_file_path = os.path.join(save_folder, "combined_test_stats.json")
        if not os.path.exists(json_file_path):
            continue
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
        r2_list.append(data["Regressor Stats"]["R2 Score"])
        corrcoef_list.append(data["Best Classifier Stats"]["Correlation Coefficient"])

    if not r2_list or not corrcoef_list:
        return

    average_r2_score = float(np.mean(r2_list))
    average_corr_score = float(np.mean(corrcoef_list))

    print(f"The average r2 score for the regressor is: {average_r2_score}")
    print(f"The average correlation CPI-AHA is: {average_corr_score}")

    results = {
        "R2 Score List": r2_list,
        "Correlation List": corrcoef_list,
        "Average R2 Score": average_r2_score,
        "Average CPI-AHA Correlation": average_corr_score,
    }

    with open(os.path.join(iterations_root, "test_results.json"), "w") as file:
        json.dump(results, file, indent=4)


def main() -> None:
    # Ensure relative paths behave like before.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"How many folds to run from a fixed {TOTAL_FOLDS}-fold CV (default: {DEFAULT_ITERATIONS}). Use 1 to run only fold 0, then rerun with higher values to compute the missing folds.",
    )
    parser.add_argument(
        "--reset-iterations",
        action="store_true",
        help="Delete the entire Iterations/ or Iterations_debug/ folder before running.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run a tiny stratified subset per fold for quick sanity checks.",
    )
    args = parser.parse_args()

    if args.iterations < 1 or args.iterations > TOTAL_FOLDS:
        raise SystemExit(f"--iterations must be between 1 and {TOTAL_FOLDS}")

    methods = ["raw"] if args.debug else ['concat', 'difference', 'ai', 'enmo', 'raw']

    metadata = pd.read_excel(os.path.join(DATA_FOLDER, "metadata2023_08.xlsx"))
    labels = metadata["hemi"].to_numpy()

    iterations_root = "Iterations_debug/" if args.debug else "Iterations/"
    if args.reset_iterations and os.path.isdir(iterations_root):
        shutil.rmtree(iterations_root)
    os.makedirs(iterations_root, exist_ok=True)

    min_mean_test_score = 0.5 if args.debug else CV_MIN_MEAN_TEST_SCORE

    rskf = RepeatedStratifiedKFold(
        n_splits=TOTAL_FOLDS,
        n_repeats=1,
        random_state=RANDOM_STATE,
    )
    splits = list(rskf.split(np.empty(metadata.shape[0]), labels))

    for iteration in range(args.iterations):
        train_indexes, test_indexes = splits[iteration]
        save_folder = os.path.join(iterations_root, f"Iteration_{iteration}") + "/"
        os.makedirs(save_folder, exist_ok=True)

        if _iteration_done(save_folder):
            print(f" ----- ITERATION {iteration} already completed; skipping ----- ")
            continue

        if args.debug:
            rng = np.random.default_rng(RANDOM_STATE + iteration)
            train_idx = np.array(train_indexes, dtype=int)
            test_idx = np.array(test_indexes, dtype=int)

            train_labels = labels[train_idx]
            test_labels = labels[test_idx]

            train_indexes = _subset_per_class(rng, train_idx, train_labels, n_per_class=2)
            test_indexes = _subset_per_class(rng, test_idx, test_labels, n_per_class=1)

        train_indexes, test_indexes = _load_or_create_iteration_split(
            iteration=iteration,
            save_folder=save_folder,
            train_indexes=np.array(train_indexes, dtype=int),
            test_indexes=np.array(test_indexes, dtype=int),
        )

        print(f" ----- ITERATION {iteration} / {args.iterations - 1} ----- ")
        _run_iteration(
            data_folder=DATA_FOLDER,
            save_folder=save_folder,
            train_indexes=train_indexes,
            test_indexes=test_indexes,
            min_mean_test_score=min_mean_test_score,
            window_size=WINDOW_SIZE,
            decimation_factor=DECIMATION_FACTOR,
            methods=methods,
        )

    _aggregate_results(iterations_root, args.iterations)

    iterations_folders = []
    for i in range(args.iterations):
        folder = os.path.join(iterations_root, f"Iteration_{i}") + "/"
        if os.path.exists(os.path.join(folder, "Week_stats", "predictions_dataframe.csv")):
            iterations_folders.append(folder)

    if len(iterations_folders) == args.iterations:
        plot_corrcoeff(iterations_folders=iterations_folders, save_folder=iterations_root)

    print(" ----- ESECUZIONE DEL MAIN TERMINATA ----- ")


if __name__ == "__main__":
    main()


"""
1. Allenare classificatori                                      (train-AHA)

2. Allenare regessore                                           (train-week)

3. Testare miglior classificatore (corfcoef) e regressore (r2)  (test-week)
    4.1 Predict samples sui best estimators (est1, est2, est3)  ->  hp_tot_list_list  
    4.2 Corrcoef tra la prima colonna di hp_tot_list_list e y
    4.3 Calcolare r2 tra regressor(hp_tot_list_list) e y

Iterazione -> corrcoef e r2 / [1, 2, 3][55, 60, 68]
Iterazione -> corrcoef e r2 / [4, 5, 6][55, 60, 68]
Iterazione -> corrcoef e r2 / [7, 8, 9][55, 60, 68]

[1, 2, 34, 4, 5, 6, 7, 8, 9]

best estimators
    est1
    est2
    est3

"""

import argparse
import json
import os
import shutil
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from explain_classifier import explain_best_classifier
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
MODEL_NAMES = ["LSTM", "GRU", "RNN", "Transformer", "CNN1D", "Reservoir"]
METHODS = [
    "raw",
    "raw_enmo_ai_jerk",
    "enmo_ai",
    "enmo_ai_jerk",
]

DEFAULT_ITERATIONS = TOTAL_FOLDS
# With f1_macro scoring, values are in [0, 1], and higher is better.
# Practical reference points (binary task):
# - Random guesser (rough baseline): ~0.50 on balanced classes (can be lower if imbalanced).
# - Decent classifier: >= 0.60.
# - Good classifier: >= 0.75.
CV_MIN_MEAN_TEST_SCORE = 0.75
DEBUG_MIN_MEAN_TEST_SCORE = 0.0


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


def _classifier_explainability_done(save_folder: str) -> bool:
    explainability_folder = os.path.join(save_folder, "Classifier_explainability")
    explainability_stats_candidates = [
        os.path.join(explainability_folder, "best_classifier_1_explainability_stats.json"),
        os.path.join(explainability_folder, "best_classifier_explainability_stats.json"),
    ]
    return any(os.path.exists(path) for path in explainability_stats_candidates)


def _week_stats_done(save_folder: str) -> bool:
    predictions_df = os.path.join(save_folder, "Week_stats", "predictions_dataframe.csv")
    return os.path.exists(predictions_df)


def _test_stats_done(save_folder: str) -> bool:
    combined_test_stats = os.path.join(save_folder, "combined_test_stats.json")
    return os.path.exists(combined_test_stats)


def _run_iteration(
    *,
    data_folder: str,
    metadata: pd.DataFrame,
    save_folder: str,
    train_indexes: Iterable[int],
    test_indexes: Iterable[int],
    min_mean_test_score: float,
    window_size: int,
    decimation_factor: int,
    methods: list,
    model_names: list,
) -> None:
    os.makedirs(save_folder, exist_ok=True)
    train_metadata = metadata.iloc[list(train_indexes)].reset_index(drop=True)
    test_metadata = metadata.iloc[list(test_indexes)].reset_index(drop=True)

    best_estimators_csv = os.path.join(save_folder, "best_estimators_results.csv")
    if not os.path.exists(best_estimators_csv):
        print(" ----- TRAINING CLASSIFIERS ----- ")
        train_select_classifiers(
            data_folder,
            save_folder=save_folder,
            metadata=train_metadata.copy(),
            l_window_size=[window_size],
            l_method=methods,
            l_decimation_factor=[decimation_factor],
            l_model_name=model_names,
        )

    print(" ----- TRAINING REGRESSOR ----- ")
    train_regressor(
        data_folder,
        save_folder=save_folder,
        metadata=train_metadata.copy(),
        min_mean_test_score=min_mean_test_score,
        window_size=window_size,
        decimation_factor=decimation_factor,
        regressor_device="cuda:2",
    )

    if not _test_stats_done(save_folder):
        print(" ----- TESTING CLASSIFIER AND REGRESSOR ----- ")
        test_classifier_regressor(
            data_folder,
            save_folder=save_folder,
            metadata=test_metadata.copy(),
            min_mean_test_score=min_mean_test_score,
            window_size=window_size,
            decimation_factor=decimation_factor,
        )

    if not _classifier_explainability_done(save_folder):
        print(" ----- EXPLAINING BEST CLASSIFIER ----- ")
        explain_best_classifier(
            data_folder,
            save_folder=save_folder,
            metadata=test_metadata.copy(),
            window_size=window_size,
            decimation_factor=decimation_factor,
        )

    if not _week_stats_done(save_folder):
        print(" ----- CREATING DASHBOARDS ----- ")
        plot_dashboards(
            data_folder,
            save_folder=save_folder,
            metadata=test_metadata.copy(),
            min_mean_test_score=min_mean_test_score,
            window_size=window_size,
            decimation_factor=decimation_factor,
        )


def _iteration_done(save_folder: str) -> bool:
    return _week_stats_done(save_folder) and _classifier_explainability_done(save_folder)


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
        r2_list.append(data["GRU Regressor Stats"]["R2 Score"])
        corrcoef_list.append(
            data["Selected Classifiers Stats"][0]["Correlation Mean Probability vs AHA"]
        )

    if not r2_list or not corrcoef_list:
        return

    average_r2_score = float(np.mean(r2_list))
    average_corr_score = float(np.mean(corrcoef_list))

    print(f"The average r2 score for the regressor is: {average_r2_score}")
    print(
        "The average correlation CPI-AHA (using the best classifier CPI for each iteration) is: "
        f"{average_corr_score}"
    )

    results = {
        "R2 Score List": r2_list,
        "Correlation List": corrcoef_list,
        "Average R2 Score": average_r2_score,
        "Average CPI-AHA Correlation (Best Classifier CPI per Iteration)": average_corr_score,
    }

    with open(os.path.join(iterations_root, "test_results.json"), "w") as file:
        json.dump(results, file, indent=4)


def _purge_prediction_caches(iterations_root: str) -> int:
    """
    Delete estimator prediction cache folders under Trained_models.
    Returns the number of removed directories.
    """
    removed = 0
    for root, dirs, _ in os.walk(iterations_root):
        if "Trained_models" not in root:
            continue
        cache_dir_name = "pred_cache"
        if cache_dir_name in dirs:
            cache_path = os.path.join(root, cache_dir_name)
            shutil.rmtree(cache_path)
            dirs.remove(cache_dir_name)
            removed += 1
    return removed


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
    parser.add_argument(
        "--purge-pred-caches",
        dest="purge_pred_caches",
        action="store_true",
        help="Delete all estimator prediction cache folders (pred_cache/) under Trained_models in the selected iterations root before running.",
    )
    args = parser.parse_args()

    if args.iterations < 1 or args.iterations > TOTAL_FOLDS:
        raise SystemExit(f"--iterations must be between 1 and {TOTAL_FOLDS}")

    methods = ["raw"] if args.debug else METHODS
    model_names = MODEL_NAMES

    metadata = pd.read_excel(
        os.path.join(DATA_FOLDER, "metadata2023_08.xlsx"),
        usecols=[
            "subject",
            "MACS",
            "AHA",
        ],
    )
    # Stratify folds by AHA quantile bins.
    labels = pd.qcut(
        metadata["AHA"],
        q=6,
        labels=False,
        duplicates="drop",
    ).to_numpy()

    iterations_root = "Iterations_debug/" if args.debug else "Iterations/"
    if args.reset_iterations and os.path.isdir(iterations_root):
        shutil.rmtree(iterations_root)
    os.makedirs(iterations_root, exist_ok=True)
    if args.purge_pred_caches:
        removed_caches = _purge_prediction_caches(iterations_root)
        print(f" ----- PURGED {removed_caches} prediction cache folder(s) ----- ")

    min_mean_test_score = DEBUG_MIN_MEAN_TEST_SCORE if args.debug else CV_MIN_MEAN_TEST_SCORE

    skf = StratifiedKFold(
        n_splits=TOTAL_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    splits = list(skf.split(np.empty(metadata.shape[0]), labels))

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
            metadata=metadata,
            save_folder=save_folder,
            train_indexes=train_indexes,
            test_indexes=test_indexes,
            min_mean_test_score=min_mean_test_score,
            window_size=WINDOW_SIZE,
            decimation_factor=DECIMATION_FACTOR,
            methods=methods,
            model_names=model_names,
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

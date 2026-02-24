import hashlib
import json
import os

import numpy as np
import joblib as jl
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from predict_samples import build_estimators_list, predict_samples
from skorch_models import make_gru_regressor_net, save_best_estimator_plots

REGRESSOR_PARAM_GRID = {
    "lr": [1e-3],
    "max_epochs": [160],
    "batch_size": [16],
    "module__bidirectional": [True],
    "module__hidden_size": [64],
    "module__num_layers": [2],
    "module__dropout": [0.0],
    "optimizer__weight_decay": [1e-4],
    "callbacks__early_stopping__patience": [25],
}


def regressor_model_path(save_folder, estimators_list):
    """Return the hash-addressed regressor model path for the current classifier set."""
    classifier_paths = sorted(
        os.path.normpath(os.path.join(str(es["estimator_dir"]), "best_estimator.joblib"))
        for es in estimators_list
    )
    payload = {
        "classifiers": classifier_paths,
        "regressor_param_grid": REGRESSOR_PARAM_GRID,
    }
    reg_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()[:10]
    return os.path.join(save_folder, "Regressors", f"regressor_{reg_hash}", "regressor.joblib")


def build_regressor_sequence(predictions_list, invalid_bitmap, window_size, decimation_factor, fs=80):
    """
    Create per-subject sequence features:
    estimator probs + invalid bit + cyclic time-of-day (sin, cos)
    """
    n_windows = np.asarray(predictions_list[0]).size
    invalid = np.asarray(invalid_bitmap, dtype=np.uint8).reshape(-1)

    probs_matrix = np.column_stack([
        np.asarray(pred, dtype=np.float32) for pred in predictions_list
    ])
    invalid_col = invalid.astype(np.float32).reshape(-1, 1)

    # ---- TIME COMPUTATION ----
    # seconds per window in the decimated signal.
    # With decimation_factor=d, effective sampling frequency is fs/d.
    seconds_per_window = window_size * decimation_factor / fs

    # absolute time in seconds from file start
    t_abs = np.arange(n_windows, dtype=np.float32) * seconds_per_window

    # seconds within the day (cyclic)
    seconds_in_day = 24 * 60 * 60
    t_day = np.mod(t_abs, seconds_in_day)

    # cyclic encoding
    angle = 2 * np.pi * t_day / seconds_in_day
    time_sin = np.sin(angle).reshape(-1, 1)
    time_cos = np.cos(angle).reshape(-1, 1)

    return np.hstack((probs_matrix, invalid_col, time_sin, time_cos)).astype(np.float32)


def _fit_regressor_with_grid_search(X, y, strat_labels, reg_dir, param_grid):
    """Run 5-fold stratified grid search, save CSV/plots, and return the best estimator."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=make_gru_regressor_net(),
        param_grid=param_grid,
        scoring="r2",
        cv=cv.split(X, strat_labels),
        refit=True,
        n_jobs=1,
        verbose=2,
    )
    grid.fit(X, y)
    print("REGRESSOR: END GRID SEARCH")
    print("REGRESSOR: best_score_ =", float(grid.best_score_))
    print("REGRESSOR: best_params_ =", grid.best_params_)

    # Persist full CV table for later inspection/comparison.
    os.makedirs(reg_dir, exist_ok=True)
    cv_results_path = os.path.join(reg_dir, "gridsearch_results.csv")

    pd.DataFrame(grid.cv_results_).sort_values(by="rank_test_score").to_csv(cv_results_path, index=False)
    # Save train/validation loss curves of the refit best estimator.
    save_best_estimator_plots(
        grid.best_estimator_,
        reg_dir,
        loss_label="MSELoss",
    )

    return grid.best_estimator_


def train_regressor(
    data_folder,
    save_folder,
    metadata,
    min_mean_test_score=None,
    window_size=None,
    decimation_factor=1,
):
    """Train (or reuse) the hash-addressed regressor built on selected classifier outputs."""
    # Select and load classifiers used to generate regressor inputs.
    best_estimators_df = pd.read_csv(
        save_folder + "best_estimators_results.csv", index_col=0
    )

    _, estimators_list = build_estimators_list(
        best_estimators_df=best_estimators_df,
        save_folder=save_folder,
        min_mean_test_score=min_mean_test_score,
        window_size=window_size,
        decimation_factor=decimation_factor,
    )

    # Hash-based destination: same classifiers + grid -> same regressor path.
    reg_model_path = regressor_model_path(
        save_folder=save_folder,
        estimators_list=estimators_list,
    )
    reg_dir = os.path.dirname(reg_model_path)
    os.makedirs(reg_dir, exist_ok=True)
    if os.path.exists(reg_model_path):
        print("REGRESSOR: already trained ->", reg_model_path)
        return

    # Build one sequence per subject from estimator probabilities.
    sequence_list = []
    for _, subject_metadata in metadata.iterrows():
        print('REGRESSOR: PATIENT ', subject_metadata['subject'], 'BEGIN')
        estimator_probs_list, _, invalid_bitmap = predict_samples(
            data_folder,
            estimators_list,
            subject_metadata,
        )
        
        sequence_list.append(build_regressor_sequence(estimator_probs_list, invalid_bitmap, window_size, decimation_factor))
        
        print('REGRESSOR: PATIENT ', subject_metadata['subject'], 'END')

    # Assemble training tensors and stratification labels.
    X = np.stack(sequence_list).astype(np.float32)
    y = metadata["AHA"].to_numpy(dtype=np.float32)
    strat_labels = (y == 100).astype(int)

    model = _fit_regressor_with_grid_search(
        X,
        y,
        strat_labels,
        reg_dir,
        REGRESSOR_PARAM_GRID,
    )
    jl.dump(model, reg_model_path)

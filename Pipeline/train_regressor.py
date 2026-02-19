import hashlib
import json
import numpy as np
import joblib as jl
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from predict_samples import build_estimators_list, predict_samples
from skorch_models import make_gru_regressor_net, save_best_estimator_plots
import os

REGRESSOR_PARAM_GRID = {
    "lr": [1e-3, 7e-4],
    "max_epochs": [160],
    "batch_size": [16],
    "module__bidirectional": [True],
    "module__hidden_size": [48, 64, 80],
    "module__num_layers": [1, 2],
    "module__dropout": [0.0, 0.1],
    "optimizer__weight_decay": [0.0, 1e-4],
    "callbacks__early_stopping__patience": [25],
}


def regressor_hash_from_estimators(estimators_list, param_grid) -> str:
    classifier_paths = sorted(
        os.path.normpath(os.path.join(str(es["estimator_dir"]), "best_estimator.joblib"))
        for es in estimators_list
    )
    payload = {
        "classifiers": classifier_paths,
        "regressor_param_grid": param_grid,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()[:10]


def regressor_model_path(save_folder, estimators_list, param_grid):
    reg_hash = regressor_hash_from_estimators(estimators_list, param_grid)
    return os.path.join(save_folder, "Regressors", f"regressor_{reg_hash}", "regressor.joblib")


def build_regressor_sequence(predictions_list, invalid_bitmap):
    n_windows = np.asarray(predictions_list[0]).size
    invalid = np.asarray(invalid_bitmap, dtype=np.uint8).reshape(-1)

    probs_matrix = np.column_stack([np.asarray(pred, dtype=np.float32) for pred in predictions_list])
    invalid_col = invalid.astype(np.float32).reshape(-1, 1)

    denom = np.float32(max(n_windows - 1, 1))
    timestamp_col = np.arange(n_windows, dtype=np.float32).reshape(-1, 1) / denom

    return np.hstack((probs_matrix, invalid_col, timestamp_col)).astype(np.float32)


def _fit_regressor_with_grid_search(X, y, strat_labels, reg_dir, param_grid):
    
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

    os.makedirs(reg_dir, exist_ok=True)
    cv_results_path = os.path.join(reg_dir, "gridsearch_results.csv")

    pd.DataFrame(grid.cv_results_).sort_values(by="rank_test_score").to_csv(cv_results_path, index=False)
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
    decimation_factor=None,
):
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

    reg_model_path = regressor_model_path(
        save_folder=save_folder,
        estimators_list=estimators_list,
        param_grid=REGRESSOR_PARAM_GRID,
    )
    reg_dir = os.path.dirname(reg_model_path)
    os.makedirs(reg_dir, exist_ok=True)
    if os.path.exists(reg_model_path):
        print("REGRESSOR: already trained ->", reg_model_path)
        return

    sequence_list = []
    for _, subject_metadata in metadata.iterrows():
        print('REGRESSOR: PATIENT ', subject_metadata['subject'], 'BEGIN')
        estimator_probs_list, _, invalid_bitmap = predict_samples(
            data_folder,
            estimators_list,
            subject_metadata,
        )
        sequence_list.append(build_regressor_sequence(estimator_probs_list, invalid_bitmap))
        print('REGRESSOR: PATIENT ', subject_metadata['subject'], 'END')

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

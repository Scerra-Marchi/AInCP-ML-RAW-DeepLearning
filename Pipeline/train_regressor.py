import hashlib
import json
import numpy as np
import joblib as jl
import pandas as pd
import torch
from torch import nn
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid
from predict_samples import build_estimators_list, predict_samples
from skorch_models import GRUSequenceClassifier
import os
import sys

REGRESSOR_SCHEMA_VERSION = "gru_seq_v3_grid5_refined"


def regressor_hash_from_estimators_specs(estimators_specs_list) -> str:
    specs = sorted(
        [
            (
                str(r["method"]),
                int(r["window_size"]),
                int(r["decimation_factor"]),
                str(r["model_type"]).split(".")[-1],
                str(r["gridsearch_hash"]),
            )
            for r in estimators_specs_list
        ]
    )
    payload = {
        "schema_version": REGRESSOR_SCHEMA_VERSION,
        "features": ["best_estimators_probs", "invalid_bit", "timestamp_norm"],
        "specs": specs,
    }
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()[:10]


def build_regressor_sequence(predictions_list, invalid_bitmap):
    if not predictions_list:
        raise ValueError("No estimator predictions were provided to build GRU regressor inputs.")

    lengths = [np.asarray(pred).size for pred in predictions_list]
    if len(set(lengths)) != 1:
        raise ValueError(f"Inconsistent prediction lengths across estimators: {lengths}")

    n_windows = lengths[0]
    invalid = np.asarray(invalid_bitmap, dtype=np.uint8).reshape(-1)
    if invalid.size != n_windows:
        raise ValueError(
            f"Invalid bitmap length ({invalid.size}) does not match prediction length ({n_windows})."
        )

    probs_matrix = np.column_stack([np.asarray(pred, dtype=np.float32) for pred in predictions_list])
    invalid_col = invalid.astype(np.float32).reshape(-1, 1)

    if n_windows <= 1:
        timestamp_col = np.zeros((n_windows, 1), dtype=np.float32)
    else:
        timestamp_col = (
            np.arange(n_windows, dtype=np.float32).reshape(-1, 1) / np.float32(n_windows - 1)
        )

    return np.hstack((probs_matrix, invalid_col, timestamp_col)).astype(np.float32)


def stack_regressor_sequences(sequence_list):
    if not sequence_list:
        raise ValueError("Cannot train GRU regressor with an empty sequence list.")

    feature_dims = [seq.shape[1] for seq in sequence_list]
    if len(set(feature_dims)) != 1:
        raise ValueError(f"Inconsistent feature dimensions across subjects: {feature_dims}")

    lengths = [seq.shape[0] for seq in sequence_list]
    min_len = min(lengths)
    if min_len < 2:
        raise ValueError(f"Sequence length too short for GRU training: min_len={min_len}")

    if len(set(lengths)) != 1:
        print(
            f"REGRESSOR: variable sequence lengths {sorted(set(lengths))}; truncating all subjects to {min_len} windows."
        )

    return np.stack([seq[:min_len] for seq in sequence_list]).astype(np.float32)


def _build_gru_regressor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return NeuralNetRegressor(
        module=GRUSequenceClassifier,
        criterion=nn.MSELoss,
        max_epochs=160,
        lr=1e-3,
        batch_size=16,
        optimizer=torch.optim.AdamW,
        optimizer__weight_decay=0.0,
        iterator_train__shuffle=True,
        train_split=False,
        module__hidden_size=64,
        module__num_layers=1,
        module__dropout=0.1,
        module__bidirectional=True,
        device=device,
        verbose=0,
    )


def _fit_gru_with_grid_search(X, y, save_folder, reg_path):
    if y.size < 2:
        model = _build_gru_regressor()
        model.fit(X, y)
        return model

    n_splits = 5 if y.size >= 10 else max(2, y.size // 2)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    base_model = _build_gru_regressor()
    # Refined search space using the observed ranking:
    # the best region was around lr=1e-3, bidirectional=True, hidden_size=64, num_layers=1.
    param_grid = {
        "lr": [1e-3, 7e-4],
        "module__bidirectional": [True],
        "module__hidden_size": [48, 64, 80],
        "module__num_layers": [1, 2],
        "module__dropout": [0.0, 0.1],
        "optimizer__weight_decay": [0.0, 1e-4],
    }

    n_candidates = len(list(ParameterGrid(param_grid)))
    scoring = "r2" if y.size >= 10 else "neg_mean_absolute_error"

    print(
        f"REGRESSOR: START GRID SEARCH ({n_splits}-fold, {n_candidates} candidates, scoring={scoring})"
    )
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        refit=True,
        n_jobs=1,
        verbose=2,
    )
    grid.fit(X, y)
    print("REGRESSOR: END GRID SEARCH")
    print("REGRESSOR: best_score_ =", float(grid.best_score_))
    print("REGRESSOR: best_params_ =", grid.best_params_)

    reg_dir = os.path.join(save_folder, "Regressors")
    os.makedirs(reg_dir, exist_ok=True)
    cv_results_path = os.path.join(reg_dir, f"{reg_path}_gridsearch_results.csv")
    best_params_path = os.path.join(reg_dir, f"{reg_path}_gridsearch_best.json")

    pd.DataFrame(grid.cv_results_).sort_values(by="rank_test_score").to_csv(cv_results_path, index=False)
    with open(best_params_path, "w") as f:
        json.dump(
            {
                "scoring": scoring,
                "best_score": float(grid.best_score_),
                "best_params": grid.best_params_,
                "cv_splits": n_splits,
                "candidates": n_candidates,
            },
            f,
            indent=4,
            default=str,
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
    best_estimators_df = pd.read_csv(save_folder + 'best_estimators_results.csv', index_col=0).sort_values(by=['mean_test_score', 'std_test_score'], ascending=False)

    # Caricamento dei classificatori

    estimators_specs_list, estimators_list = build_estimators_list(
        best_estimators_df=best_estimators_df,
        save_folder=save_folder,
        min_mean_test_score=min_mean_test_score,
        window_size=window_size,
        decimation_factor=decimation_factor,
    )

    reg_path = 'regressor_' + regressor_hash_from_estimators_specs(estimators_specs_list)
    os.makedirs(save_folder + 'Regressors/', exist_ok = True)
    reg_full_path = save_folder + 'Regressors/' + reg_path
    if os.path.exists(reg_full_path):
        print("REGRESSOR: already trained ->", reg_full_path)
        return

    # Allenamento del regressore

    sequence_list = []
    y_targets = []

    for _, subject_metadata in metadata.iterrows():
        print('REGRESSOR: PATIENT ', subject_metadata['subject'], 'BEGIN')
        y_list, _, invalid_bitmap = predict_samples(
            data_folder,
            estimators_list,
            subject_metadata,
        )
        sequence_list.append(build_regressor_sequence(y_list, invalid_bitmap))
        y_targets.append(float(subject_metadata["AHA"]))
        print('REGRESSOR: PATIENT ', subject_metadata['subject'], 'END')
        sys.stdout.flush()
        sys.stderr.flush()

    X = stack_regressor_sequences(sequence_list)
    y = np.asarray(y_targets, dtype=np.float32)

    model = _fit_gru_with_grid_search(X, y, save_folder, reg_path)
    jl.dump(model, reg_full_path)

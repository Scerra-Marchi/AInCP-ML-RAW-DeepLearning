import hashlib
import json
import os
from tempfile import TemporaryDirectory

import joblib as jl
import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from predict_samples import build_estimators_list, predict_samples
from read_file import read_file
from signal_features import compute_window_sensor_features
from skorch_models import (
    GRUSequenceRegressor,
    SequenceStandardScaler,
    make_regressor_net,
    save_best_estimator_plots,
    set_global_determinism,
)

POSITIVE_THRESHOLD = 0.5
FOUR_MINUTES_BLOCK_SECONDS = 4 * 60
HOUR_BLOCK_SECONDS = 3600
WHOLE_WEEK_BLOCK_SECONDS = 6 * 24 * HOUR_BLOCK_SECONDS
RAW_WINDOW_STD_TOL = 0.005
SENSOR_FEATURE_EPS = 1e-6
SENSOR_WINDOW_FEATURE_KEYS = (
    "enmo_mean_d",
    "enmo_mean_nd",
    "enmo_diff",
    "enmo_log_ratio",
    "signed_ai_enmo",
    "bilateral_enmo_mean",
    "jerk_mean_d",
    "jerk_mean_nd",
)
PREPROCESSING_CONFIG = {
    "positive_threshold": POSITIVE_THRESHOLD,
    "gru_sequence_scaling": True,
    "block_sensor_features": list(SENSOR_WINDOW_FEATURE_KEYS),
}

GRU_MODEL_PARAM_GRID = {
    "model__lr": [0.1, 0.2],
    "model__max_epochs": [500],
    "model__batch_size": [128],
    "model__module__hidden_size": [128, 256],
    "model__module__num_layers": [1, 2],
    "model__module__dropout": [0.1, 0.25],
    "model__optimizer__weight_decay": [0.0, 1e-3],
    "model__callbacks__early_stopping__patience": [25],
}
GRU_PARAM_GRID = [
    {
        **GRU_MODEL_PARAM_GRID,
        "prep__block_seconds": [
            FOUR_MINUTES_BLOCK_SECONDS,
            HOUR_BLOCK_SECONDS,
            24 * HOUR_BLOCK_SECONDS,
            WHOLE_WEEK_BLOCK_SECONDS,
        ],
    },
]


def _classifier_model_paths(estimators_list):
    return sorted(
        os.path.normpath(os.path.join(str(es["estimator_dir"]), "best_estimator.joblib"))
        for es in estimators_list
    )


def _hash_payload(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()[:10]


def _hashed_model_path(save_folder, folder_prefix, filename, estimators_list, model_payload):
    payload = {
        "classifiers": _classifier_model_paths(estimators_list),
        "preprocessing_config": PREPROCESSING_CONFIG,
        **model_payload,
    }
    model_hash = _hash_payload(payload)
    return os.path.join(save_folder, "Regressors", f"{folder_prefix}_{model_hash}", filename)


def regressor_model_path(save_folder, estimators_list):
    return _hashed_model_path(
        save_folder=save_folder,
        folder_prefix="gru_regressor",
        filename="regressor.joblib",
        estimators_list=estimators_list,
        model_payload={"gru_param_grid": GRU_PARAM_GRID},
    )


def build_block_feature_names(n_features, n_estimators):
    names = [f"classifier_{i}_block_mean" for i in range(n_estimators)]
    names += [f"classifier_{i}_block_std" for i in range(n_estimators)]
    names += [f"classifier_{i}_block_q25" for i in range(n_estimators)]
    names += [f"classifier_{i}_block_q75" for i in range(n_estimators)]
    names += [f"classifier_{i}_block_posrate" for i in range(n_estimators)]
    names += [
        "block_enmo_mean_d",
        "block_enmo_mean_nd",
        "block_enmo_diff",
        "block_enmo_log_ratio",
        "block_signed_ai_enmo",
        "block_bilateral_enmo_mean",
        "block_jerk_mean_d",
        "block_jerk_mean_nd",
    ]
    names += ["block_valid_fraction", "time_sin", "time_cos"]
    return names[:n_features]


def _read_windowed_raw_signals(data_folder, subject_metadata, window_size, decimation_factor, input_type="week"):
    subject = int(subject_metadata["subject"])
    D, ND = read_file(
        data_folder=data_folder,
        subject=subject,
        WINDOW_SIZE=window_size,
        decimation_factor=decimation_factor,
        input_type=input_type,
    )

    n_windows = len(D) // window_size
    D_w = D.reshape(n_windows, window_size, 3)
    ND_w = ND.reshape(n_windows, window_size, 3)
    return D_w, ND_w


def _compute_sensor_window_features(data_folder, subject_metadata, window_size, decimation_factor, input_type="week"):
    D_w, ND_w = _read_windowed_raw_signals(
        data_folder,
        subject_metadata,
        window_size,
        decimation_factor,
        input_type=input_type,
    )
    return compute_window_sensor_features(
        D_w,
        ND_w,
        epsilon=SENSOR_FEATURE_EPS,
        std_tol=RAW_WINDOW_STD_TOL,
    )


def _sensor_feature_matrix(sensor_window_features):
    return np.column_stack(
        [np.asarray(sensor_window_features[key], dtype=np.float32) for key in SENSOR_WINDOW_FEATURE_KEYS]
    )


def build_regressor_sample(
    data_folder,
    estimators_list,
    subject_metadata,
    *,
    input_type="week",
    predictions_list=None,
    invalid_bitmap=None,
):
    if predictions_list is None or invalid_bitmap is None:
        predictions_list, _, invalid_bitmap = predict_samples(
            data_folder,
            estimators_list,
            subject_metadata,
        )
    sensor_window_features, sensor_invalid_bitmap = _compute_sensor_window_features(
        data_folder,
        subject_metadata,
        estimators_list[0]["window_size"],
        estimators_list[0]["decimation_factor"],
        input_type=input_type,
    )
    invalid_bitmap = np.asarray(invalid_bitmap, dtype=np.uint8)
    if invalid_bitmap.shape != sensor_invalid_bitmap.shape or not np.array_equal(
        invalid_bitmap,
        sensor_invalid_bitmap,
    ):
        raise ValueError("Sensor-derived window features are misaligned with predict_samples invalid_bitmap.")

    sample = {
        "predictions_list": [np.asarray(pred, dtype=np.float32) for pred in predictions_list],
        "invalid_bitmap": invalid_bitmap,
        "sensor_window_matrix": _sensor_feature_matrix(sensor_window_features),
    }
    return sample

def _build_block_sequence(
    probs_matrix,
    invalid,
    sensor_window_matrix,
    window_size,
    decimation_factor,
    fs,
    block_seconds,
):
    n_windows, n_estimators = probs_matrix.shape
    valid_mask = ~invalid.astype(bool)
    seconds_per_window = window_size * decimation_factor / fs
    steps_per_block = max(1, int(round(block_seconds / seconds_per_window)))
    n_blocks = int(np.ceil(n_windows / steps_per_block))

    block_mean = np.full((n_blocks, n_estimators), 0.5, dtype=np.float32)
    block_std = np.zeros((n_blocks, n_estimators), dtype=np.float32)
    block_q25 = np.full((n_blocks, n_estimators), 0.5, dtype=np.float32)
    block_q75 = np.full((n_blocks, n_estimators), 0.5, dtype=np.float32)
    block_pos = np.full((n_blocks, n_estimators), 0.5, dtype=np.float32)
    block_sensor = np.zeros((n_blocks, len(SENSOR_WINDOW_FEATURE_KEYS)), dtype=np.float32)
    block_valid_fraction = np.zeros((n_blocks, 1), dtype=np.float32)
    block_center_seconds = np.zeros(n_blocks, dtype=np.float32)
    sensor_matrix = np.asarray(sensor_window_matrix, dtype=np.float32)

    for b in range(n_blocks):
        start = b * steps_per_block
        end = min(n_windows, (b + 1) * steps_per_block)
        block_probs = probs_matrix[start:end]
        block_valid = valid_mask[start:end]
        block_sensor_values = sensor_matrix[start:end]

        if block_valid.size > 0:
            block_valid_fraction[b, 0] = float(block_valid.mean())

        block_center_seconds[b] = ((start + end - 1) * 0.5) * seconds_per_window

        if np.any(block_valid):
            # Vectorized per-estimator stats on valid windows only.
            valid_vals = np.where(block_valid[:, None], block_probs, np.nan)
            block_mean[b] = np.nanmean(valid_vals, axis=0).astype(np.float32)
            block_std[b] = np.nanstd(valid_vals, axis=0).astype(np.float32)
            block_q25[b] = np.nanquantile(valid_vals, 0.25, axis=0).astype(np.float32)
            block_q75[b] = np.nanquantile(valid_vals, 0.75, axis=0).astype(np.float32)
            block_pos[b] = np.nanmean(
                np.where(block_valid[:, None], block_probs >= POSITIVE_THRESHOLD, np.nan),
                axis=0,
            ).astype(np.float32)
            block_sensor[b] = np.nanmean(
                np.where(block_valid[:, None], block_sensor_values, np.nan),
                axis=0,
            ).astype(np.float32)

    seconds_in_day = 24 * 60 * 60
    angle = 2 * np.pi * np.mod(block_center_seconds, seconds_in_day) / seconds_in_day
    time_sin = np.sin(angle).reshape(-1, 1).astype(np.float32)
    time_cos = np.cos(angle).reshape(-1, 1).astype(np.float32)

    return np.hstack(
        (
            block_mean,
            block_std,
            block_q25,
            block_q75,
            block_pos,
            block_sensor,
            block_valid_fraction,
            time_sin,
            time_cos,
        )
    ).astype(np.float32)

def build_block_regressor_sequence(
    predictions_list,
    invalid_bitmap,
    sensor_window_matrix,
    window_size,
    decimation_factor,
    fs=80,
    block_seconds=HOUR_BLOCK_SECONDS,
):
    probs_matrix = np.column_stack([np.asarray(pred, dtype=np.float32) for pred in predictions_list])
    invalid = np.asarray(invalid_bitmap, dtype=np.uint8).reshape(-1)
    return _build_block_sequence(
        probs_matrix,
        invalid,
        sensor_window_matrix,
        window_size,
        decimation_factor,
        fs,
        block_seconds,
    )


class BlockSequencePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        window_size,
        decimation_factor,
        fs=80,
        block_seconds=HOUR_BLOCK_SECONDS,
    ):
        self.window_size = window_size
        self.decimation_factor = decimation_factor
        self.fs = fs
        self.mode = "block"
        self.block_seconds = block_seconds

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        sequences = [
            build_block_regressor_sequence(
                sample["predictions_list"],
                sample["invalid_bitmap"],
                sample["sensor_window_matrix"],
                self.window_size,
                self.decimation_factor,
                fs=self.fs,
                block_seconds=self.block_seconds,
            )
            for sample in X
        ]
        return np.stack(sequences).astype(np.float32)

def _fit_pipeline_with_grid_search(
    X_raw,
    y,
    strat_labels,
    model_dir,
    estimator,
    param_grid,
    model_label,
    loss_label,
):
    set_global_determinism()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"{model_label}: START GRID SEARCH")
    with TemporaryDirectory(prefix="pipeline_cache_", dir=model_dir) as cache_dir:
        estimator.set_params(memory=Memory(location=cache_dir, verbose=0))
        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="r2",
            cv=cv.split(X_raw, strat_labels),
            refit=True,
            n_jobs=1,
            return_train_score=True,
            verbose=1,
        )
        grid.fit(X_raw, y)
        best_estimator = grid.best_estimator_
        best_estimator.set_params(memory=None)
        cv_results = pd.DataFrame(grid.cv_results_)

    print(f"{model_label}: END GRID SEARCH")
    print(f"{model_label}: best_score_ =", float(grid.best_score_))
    print(f"{model_label}: best_params_ =", grid.best_params_)

    os.makedirs(model_dir, exist_ok=True)
    cv_results.sort_values(by="rank_test_score").to_csv(
        os.path.join(model_dir, "gridsearch_results.csv"),
        index=False,
    )
    save_best_estimator_plots(
        best_estimator.named_steps["model"],
        model_dir,
        loss_label=loss_label,
    )
    return best_estimator


def _build_gru_pipeline(window_size, decimation_factor, regressor_device):
    return Pipeline(
        [
            (
                "prep",
                BlockSequencePreprocessor(
                    window_size=window_size,
                    decimation_factor=decimation_factor,
                ),
            ),
            ("scaler", SequenceStandardScaler()),
            (
                "model",
                make_regressor_net(
                    module=GRUSequenceRegressor,
                    module_kwargs={
                        "return_last_step": True,
                    },
                    device=regressor_device,
                ),
            ),
        ]
    )

def train_regressor(
    data_folder,
    save_folder,
    metadata,
    min_mean_test_score=None,
    window_size=None,
    decimation_factor=1,
    regressor_device="cuda:2",
):
    best_estimators_df = pd.read_csv(save_folder + "best_estimators_results.csv", index_col=0)
    _, estimators_list = build_estimators_list(
        best_estimators_df=best_estimators_df,
        save_folder=save_folder,
        min_mean_test_score=min_mean_test_score,
        window_size=window_size,
        decimation_factor=decimation_factor,
    )

    gru_model_path = regressor_model_path(save_folder=save_folder, estimators_list=estimators_list)
    gru_dir = os.path.dirname(gru_model_path)

    os.makedirs(gru_dir, exist_ok=True)
    if os.path.exists(gru_model_path):
        print("GRU REGRESSOR: already trained ->", gru_model_path)
        return

    raw_samples = []
    for _, subject_metadata in metadata.iterrows():
        print("REGRESSOR: PATIENT", subject_metadata["subject"], "BEGIN")
        raw_samples.append(
            build_regressor_sample(
                data_folder,
                estimators_list,
                subject_metadata,
                input_type="week",
            )
        )
        print("REGRESSOR: PATIENT", subject_metadata["subject"], "END")

    X_raw = np.asarray(raw_samples, dtype=object)
    y = metadata["AHA"].to_numpy(dtype=np.float32).reshape(-1, 1)
    strat_labels = pd.qcut(
        metadata["AHA"],
        q=6,
        labels=False,
        duplicates="drop",
    ).to_numpy()

    gru_model = _fit_pipeline_with_grid_search(
        X_raw=X_raw,
        y=y,
        strat_labels=strat_labels,
        model_dir=gru_dir,
        estimator=_build_gru_pipeline(
            window_size,
            decimation_factor,
            regressor_device,
        ),
        param_grid=GRU_PARAM_GRID,
        model_label="GRU REGRESSOR",
        loss_label="MSELoss",
    )
    jl.dump(gru_model, gru_model_path)

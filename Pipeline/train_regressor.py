import hashlib
import json
import os

import joblib as jl
import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from predict_samples import build_estimators_list, predict_samples
from skorch_models import (
    FFNNRegressor,
    GRUSequenceRegressor,
    make_regressor_net,
    save_best_estimator_plots,
    set_global_determinism,
)

CUMULATIVE_POSITIVE_THRESHOLD = 0.5
CUMULATIVE_QUANTILE_HIST_BINS = 128
HOUR_BLOCK_SECONDS = 3600
PREPROCESSING_CONFIG = {
    "cumulative_positive_threshold": CUMULATIVE_POSITIVE_THRESHOLD,
    "cumulative_quantile_hist_bins": CUMULATIVE_QUANTILE_HIST_BINS,
    "gru_sequence_scaling": True,
}

GRU_MODEL_PARAM_GRID = {
    "model__lr": [1e-1, 1e-2],
    "model__max_epochs": [500],
    "model__batch_size": [128],
    "model__module__hidden_size": [16, 32],
    "model__module__num_layers": [1, 2],
    "model__module__dropout": [0.2],
    "model__optimizer__weight_decay": [0.0, 1e-4],
    "model__callbacks__early_stopping__patience": [25],
}
GRU_PARAM_GRID = [
    {**GRU_MODEL_PARAM_GRID, "prep__mode": ["raw"]},
    # {**GRU_MODEL_PARAM_GRID, "prep__mode": ["cumulative"]},
    {
        **GRU_MODEL_PARAM_GRID,
        "prep__mode": ["hourly"],
        "prep__hour_block_seconds": [
            HOUR_BLOCK_SECONDS,
            24 * HOUR_BLOCK_SECONDS,
        ],
    },
]

FFNN_PREPROCESSING_MODE = "final_stats"
FFNN_PARAM_GRID = {
    "model__lr": [1e-1, 1e-2, 1e-3],
    "model__max_epochs": [500],
    "model__batch_size": [128],
    "model__module__hidden_sizes": [(), (64,), (128, 64)],
    "model__module__dropout": [0.0, 0.1, 0.2],
    "model__optimizer__weight_decay": [0.0, 1e-2, 1e-3],
    "model__callbacks__early_stopping__patience": [10],
}


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


def ffnn_model_path(save_folder, estimators_list):
    return _hashed_model_path(
        save_folder=save_folder,
        folder_prefix="ffnn_regressor",
        filename="ffnn.joblib",
        estimators_list=estimators_list,
        model_payload={
            "ffnn_preprocessing_mode": FFNN_PREPROCESSING_MODE,
            "ffnn_param_grid": FFNN_PARAM_GRID,
        },
    )


def _raw_feature_names(n_estimators):
    names = [f"classifier_{i}" for i in range(n_estimators)]
    names += ["invalid_flag", "time_sin", "time_cos"]
    return names


def _cumulative_feature_names(n_estimators):
    names = [f"classifier_{i}" for i in range(n_estimators)]
    names += [f"classifier_{i}_cummean" for i in range(n_estimators)]
    names += [f"classifier_{i}_cumstd" for i in range(n_estimators)]
    names += [f"classifier_{i}_cumq25" for i in range(n_estimators)]
    names += [f"classifier_{i}_cumq75" for i in range(n_estimators)]
    names += [f"classifier_{i}_cumposrate" for i in range(n_estimators)]
    names += ["invalid_flag", "cum_valid_fraction", "time_sin", "time_cos"]
    return names


def _hourly_feature_names(n_estimators):
    names = [f"classifier_{i}_hourmean" for i in range(n_estimators)]
    names += [f"classifier_{i}_hourstd" for i in range(n_estimators)]
    names += [f"classifier_{i}_hourq25" for i in range(n_estimators)]
    names += [f"classifier_{i}_hourq75" for i in range(n_estimators)]
    names += [f"classifier_{i}_hourposrate" for i in range(n_estimators)]
    names += ["hour_valid_fraction", "time_sin", "time_cos"]
    return names


def _final_stats_feature_names(n_estimators):
    names = [f"classifier_{i}_cummean" for i in range(n_estimators)]
    names += [f"classifier_{i}_cumstd" for i in range(n_estimators)]
    names += [f"classifier_{i}_cumq25" for i in range(n_estimators)]
    names += [f"classifier_{i}_cumq75" for i in range(n_estimators)]
    names += [f"classifier_{i}_cumposrate" for i in range(n_estimators)]
    names += ["cum_valid_fraction"]
    return names


_FEATURE_NAME_BUILDERS = {
    "raw": _raw_feature_names,
    "cumulative": _cumulative_feature_names,
    "hourly": _hourly_feature_names,
    "final_stats": _final_stats_feature_names,
}


def build_regressor_feature_names(mode, n_features, n_estimators):
    return _FEATURE_NAME_BUILDERS[mode](n_estimators)[:n_features]


def _time_features(n_steps, window_size, decimation_factor, fs):
    seconds_per_window = window_size * decimation_factor / fs
    t_abs = np.arange(n_steps, dtype=np.float32) * seconds_per_window
    seconds_in_day = 24 * 60 * 60
    angle = 2 * np.pi * np.mod(t_abs, seconds_in_day) / seconds_in_day
    return np.sin(angle).reshape(-1, 1), np.cos(angle).reshape(-1, 1)


def _compute_cumulative_quantiles_histogram(probs_matrix, valid_mask, n_bins):
    n_windows, n_estimators = probs_matrix.shape
    cum_q25 = np.full((n_windows, n_estimators), 0.5, dtype=np.float32)
    cum_q75 = np.full((n_windows, n_estimators), 0.5, dtype=np.float32)
    valid_int = valid_mask.astype(np.int32)
    idx_rows = np.arange(n_windows)

    for est_idx in range(n_estimators):
        values = probs_matrix[:, est_idx]
        bin_idx = np.clip((values * n_bins).astype(np.int32), 0, n_bins - 1)
        hist = np.zeros((n_windows, n_bins), dtype=np.int32)
        hist[idx_rows, bin_idx] = valid_int
        cum_hist = np.cumsum(hist, axis=0)

        counts = cum_hist.sum(axis=1)
        has_counts = counts > 0
        if not np.any(has_counts):
            continue

        cum_hist_valid = cum_hist[has_counts]
        q25_target = np.ceil(0.25 * counts[has_counts]).astype(np.int32)
        q75_target = np.ceil(0.75 * counts[has_counts]).astype(np.int32)
        q25_bins = (cum_hist_valid >= q25_target[:, None]).argmax(axis=1)
        q75_bins = (cum_hist_valid >= q75_target[:, None]).argmax(axis=1)

        cum_q25[has_counts, est_idx] = (q25_bins.astype(np.float32) + 0.5) / float(n_bins)
        cum_q75[has_counts, est_idx] = (q75_bins.astype(np.float32) + 0.5) / float(n_bins)

    return cum_q25, cum_q75


def _compute_cumulative_summary_features(probs_matrix, valid_mask):
    n_windows, n_estimators = probs_matrix.shape
    valid_col = valid_mask.astype(np.float32).reshape(-1, 1)
    valid_probs = probs_matrix * valid_col

    cum_count = np.cumsum(valid_col, axis=0)
    cum_sum = np.cumsum(valid_probs, axis=0)
    cum_sq_sum = np.cumsum(valid_probs * valid_probs, axis=0)
    cum_pos_sum = np.cumsum((probs_matrix >= CUMULATIVE_POSITIVE_THRESHOLD).astype(np.float32) * valid_col, axis=0)

    cum_mean = np.full((n_windows, n_estimators), 0.5, dtype=np.float32)
    np.divide(cum_sum, cum_count, out=cum_mean, where=cum_count > 0)

    second_moment = np.zeros((n_windows, n_estimators), dtype=np.float32)
    np.divide(cum_sq_sum, cum_count, out=second_moment, where=cum_count > 0)
    cum_var = np.maximum(second_moment - cum_mean * cum_mean, 0.0)
    cum_std = np.sqrt(cum_var, dtype=np.float32)
    cum_std[cum_count[:, 0] == 0] = 0.0

    cum_pos_rate = np.full((n_windows, n_estimators), 0.5, dtype=np.float32)
    np.divide(cum_pos_sum, cum_count, out=cum_pos_rate, where=cum_count > 0)

    cum_q25, cum_q75 = _compute_cumulative_quantiles_histogram(
        probs_matrix,
        valid_mask,
        n_bins=CUMULATIVE_QUANTILE_HIST_BINS,
    )

    valid_fraction = cum_count / np.arange(1, n_windows + 1, dtype=np.float32).reshape(-1, 1)
    return cum_mean, cum_std, cum_q25, cum_q75, cum_pos_rate, valid_fraction


def _compute_final_quantiles_histogram(probs_matrix, valid_mask, n_bins):
    _, n_estimators = probs_matrix.shape
    q25 = np.full(n_estimators, 0.5, dtype=np.float32)
    q75 = np.full(n_estimators, 0.5, dtype=np.float32)
    if not np.any(valid_mask):
        return q25, q75

    valid_probs = probs_matrix[valid_mask]
    counts = np.ones(valid_probs.shape[0], dtype=np.int32)
    for est_idx in range(n_estimators):
        values = valid_probs[:, est_idx]
        bin_idx = np.clip((values * n_bins).astype(np.int32), 0, n_bins - 1)
        hist = np.bincount(bin_idx, weights=counts, minlength=n_bins)
        total = int(hist.sum())
        q25_target = int(np.ceil(0.25 * total))
        q75_target = int(np.ceil(0.75 * total))
        cum_hist = np.cumsum(hist)
        q25_bin = int((cum_hist >= q25_target).argmax())
        q75_bin = int((cum_hist >= q75_target).argmax())
        q25[est_idx] = (q25_bin + 0.5) / float(n_bins)
        q75[est_idx] = (q75_bin + 0.5) / float(n_bins)
    return q25, q75


def _compute_final_summary_features(probs_matrix, valid_mask):
    n_windows, n_estimators = probs_matrix.shape
    if n_windows == 0 or not np.any(valid_mask):
        return np.hstack(
            (
                np.full(n_estimators, 0.5, dtype=np.float32),
                np.zeros(n_estimators, dtype=np.float32),
                np.full(n_estimators, 0.5, dtype=np.float32),
                np.full(n_estimators, 0.5, dtype=np.float32),
                np.full(n_estimators, 0.5, dtype=np.float32),
                np.array([0.0], dtype=np.float32),
            )
        )

    valid_vals = np.where(valid_mask[:, None], probs_matrix, np.nan)
    mean = np.nanmean(valid_vals, axis=0).astype(np.float32)
    std = np.nanstd(valid_vals, axis=0).astype(np.float32)
    pos_rate = np.nanmean(
        np.where(valid_mask[:, None], probs_matrix >= CUMULATIVE_POSITIVE_THRESHOLD, np.nan),
        axis=0,
    ).astype(np.float32)

    q25, q75 = _compute_final_quantiles_histogram(
        probs_matrix,
        valid_mask,
        n_bins=CUMULATIVE_QUANTILE_HIST_BINS,
    )

    valid_fraction = np.array([valid_mask.mean()], dtype=np.float32)
    return np.hstack((mean, std, q25, q75, pos_rate, valid_fraction))


def _build_raw_sequence(probs_matrix, invalid, window_size, decimation_factor, fs, hour_block_seconds):
    del hour_block_seconds
    time_sin, time_cos = _time_features(probs_matrix.shape[0], window_size, decimation_factor, fs)
    invalid_col = invalid.astype(np.float32).reshape(-1, 1)
    return np.hstack((probs_matrix, invalid_col, time_sin, time_cos)).astype(np.float32)


def _build_cumulative_sequence(probs_matrix, invalid, window_size, decimation_factor, fs, hour_block_seconds):
    del hour_block_seconds
    valid_mask = ~invalid.astype(bool)
    (
        cum_mean,
        cum_std,
        cum_q25,
        cum_q75,
        cum_pos_rate,
        valid_fraction,
    ) = _compute_cumulative_summary_features(probs_matrix, valid_mask)
    time_sin, time_cos = _time_features(probs_matrix.shape[0], window_size, decimation_factor, fs)
    invalid_col = invalid.astype(np.float32).reshape(-1, 1)
    return np.hstack(
        (
            probs_matrix,
            cum_mean,
            cum_std,
            cum_q25,
            cum_q75,
            cum_pos_rate,
            invalid_col,
            valid_fraction,
            time_sin,
            time_cos,
        )
    ).astype(np.float32)


def _build_hourly_sequence(
    probs_matrix,
    invalid,
    window_size,
    decimation_factor,
    fs,
    hour_block_seconds,
):
    n_windows, n_estimators = probs_matrix.shape
    valid_mask = ~invalid.astype(bool)
    seconds_per_window = window_size * decimation_factor / fs
    steps_per_block = max(1, int(round(hour_block_seconds / seconds_per_window)))
    n_blocks = int(np.ceil(n_windows / steps_per_block))

    block_mean = np.full((n_blocks, n_estimators), 0.5, dtype=np.float32)
    block_std = np.zeros((n_blocks, n_estimators), dtype=np.float32)
    block_q25 = np.full((n_blocks, n_estimators), 0.5, dtype=np.float32)
    block_q75 = np.full((n_blocks, n_estimators), 0.5, dtype=np.float32)
    block_pos = np.full((n_blocks, n_estimators), 0.5, dtype=np.float32)
    block_valid_fraction = np.zeros((n_blocks, 1), dtype=np.float32)
    block_center_seconds = np.zeros(n_blocks, dtype=np.float32)

    for b in range(n_blocks):
        start = b * steps_per_block
        end = min(n_windows, (b + 1) * steps_per_block)
        block_probs = probs_matrix[start:end]
        block_valid = valid_mask[start:end]

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
                np.where(block_valid[:, None], block_probs >= CUMULATIVE_POSITIVE_THRESHOLD, np.nan),
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
            block_valid_fraction,
            time_sin,
            time_cos,
        )
    ).astype(np.float32)


_SEQUENCE_BUILDERS = {
    "raw": _build_raw_sequence,
    "cumulative": _build_cumulative_sequence,
    "hourly": _build_hourly_sequence,
}


def build_regressor_sequence(
    predictions_list,
    invalid_bitmap,
    window_size,
    decimation_factor,
    fs=80,
    mode="cumulative",
    hour_block_seconds=HOUR_BLOCK_SECONDS,
):
    probs_matrix = np.column_stack([np.asarray(pred, dtype=np.float32) for pred in predictions_list])
    invalid = np.asarray(invalid_bitmap, dtype=np.uint8).reshape(-1)
    return _SEQUENCE_BUILDERS[mode](
        probs_matrix,
        invalid,
        window_size,
        decimation_factor,
        fs,
        hour_block_seconds,
    )


class SequencePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        window_size,
        decimation_factor,
        fs=80,
        mode="cumulative",
        hour_block_seconds=HOUR_BLOCK_SECONDS,
    ):
        self.window_size = window_size
        self.decimation_factor = decimation_factor
        self.fs = fs
        self.mode = mode
        self.hour_block_seconds = hour_block_seconds

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        sequences = [
            build_regressor_sequence(
                sample["predictions_list"],
                sample["invalid_bitmap"],
                self.window_size,
                self.decimation_factor,
                fs=self.fs,
                mode=self.mode,
                hour_block_seconds=self.hour_block_seconds,
            )
            for sample in X
        ]
        return np.stack(sequences).astype(np.float32)


class SequenceStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        self.scaler.fit(X.reshape(-1, X.shape[-1]))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1]))
        return X_scaled.reshape(X.shape).astype(np.float32)


class FinalStatsPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for sample in X:
            probs_matrix = np.column_stack(
                [np.asarray(pred, dtype=np.float32) for pred in sample["predictions_list"]]
            )
            valid_mask = ~np.asarray(sample["invalid_bitmap"], dtype=np.uint8).reshape(-1).astype(bool)
            features.append(_compute_final_summary_features(probs_matrix, valid_mask))
        return np.stack(features).astype(np.float32)


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
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="r2",
        cv=cv.split(X_raw, strat_labels),
        refit=True,
        n_jobs=1,
        return_train_score=True,
        verbose=0,
    )
    grid.fit(X_raw, y)
    print(f"{model_label}: END GRID SEARCH")
    print(f"{model_label}: best_score_ =", float(grid.best_score_))
    print(f"{model_label}: best_params_ =", grid.best_params_)

    os.makedirs(model_dir, exist_ok=True)
    pd.DataFrame(grid.cv_results_).sort_values(by="rank_test_score").to_csv(
        os.path.join(model_dir, "gridsearch_results.csv"),
        index=False,
    )
    save_best_estimator_plots(
        grid.best_estimator_.named_steps["model"],
        model_dir,
        loss_label=loss_label,
    )
    return grid.best_estimator_


def _build_gru_pipeline(model_dir, window_size, decimation_factor, regressor_device):
    return Pipeline(
        [
            (
                "prep",
                SequencePreprocessor(
                    window_size=window_size,
                    decimation_factor=decimation_factor,
                    mode="cumulative",
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
        ],
        memory=Memory(location=os.path.join(model_dir, "pipeline_cache"), verbose=0),
    )


def _build_ffnn_pipeline(model_dir, window_size, decimation_factor, regressor_device):
    del window_size, decimation_factor
    return Pipeline(
        [
            ("prep", FinalStatsPreprocessor()),
            ("scaler", StandardScaler()),
            ("model", make_regressor_net(module=FFNNRegressor, device=regressor_device)),
        ],
        memory=Memory(location=os.path.join(model_dir, "pipeline_cache"), verbose=0),
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
    ffnn_path = ffnn_model_path(save_folder=save_folder, estimators_list=estimators_list)
    gru_dir = os.path.dirname(gru_model_path)
    ffnn_dir = os.path.dirname(ffnn_path)

    os.makedirs(gru_dir, exist_ok=True)
    os.makedirs(ffnn_dir, exist_ok=True)
    if os.path.exists(gru_model_path) and os.path.exists(ffnn_path):
        print("GRU REGRESSOR: already trained ->", gru_model_path)
        print("FFNN REGRESSOR: already trained ->", ffnn_path)
        return

    raw_samples = []
    for _, subject_metadata in metadata.iterrows():
        print("REGRESSOR: PATIENT", subject_metadata["subject"], "BEGIN")
        estimator_probs_list, _, invalid_bitmap = predict_samples(
            data_folder,
            estimators_list,
            subject_metadata,
        )
        raw_samples.append(
            {
                "predictions_list": [np.asarray(pred, dtype=np.float32) for pred in estimator_probs_list],
                "invalid_bitmap": np.asarray(invalid_bitmap, dtype=np.uint8),
            }
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

    if not os.path.exists(ffnn_path):
        ffnn_model = _fit_pipeline_with_grid_search(
            X_raw=X_raw,
            y=y,
            strat_labels=strat_labels,
            model_dir=ffnn_dir,
            estimator=_build_ffnn_pipeline(
                ffnn_dir,
                window_size,
                decimation_factor,
                regressor_device,
            ),
            param_grid=FFNN_PARAM_GRID,
            model_label="FFNN REGRESSOR",
            loss_label="MSELoss",
        )
        jl.dump(ffnn_model, ffnn_path)

    if not os.path.exists(gru_model_path):
        gru_model = _fit_pipeline_with_grid_search(
            X_raw=X_raw,
            y=y,
            strat_labels=strat_labels,
            model_dir=gru_dir,
            estimator=_build_gru_pipeline(
                gru_dir,
                window_size,
                decimation_factor,
                regressor_device,
            ),
            param_grid=GRU_PARAM_GRID,
            model_label="GRU REGRESSOR",
            loss_label="MSELoss",
        )
        jl.dump(gru_model, gru_model_path)

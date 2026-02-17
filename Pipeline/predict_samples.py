import numpy as np
import joblib as jl
import os
import pandas as pd
import ray
import torch
from create_windows import create_windows


# Resource requests are set at submission time via `.options(...)` below.
@ray.remote
def _predict_proba_remote(estimator, X):
    return estimator.predict_proba(X)[:, 1]


def build_estimators_list(best_estimators_df, save_folder, min_mean_test_score, window_size, decimation_factor):
    """
    Build the estimator dictionaries consumed by predict_samples().
    Each estimator carries its own cache directory under the model folder.
    """
    estimators_specs_list = [
        row
        for _, row in best_estimators_df[
            (best_estimators_df["mean_test_score"] >= min_mean_test_score)
            & (best_estimators_df["window_size"] == window_size)
            & (best_estimators_df["decimation_factor"] == decimation_factor)
        ].iterrows()
    ]

    estimators_list = []

    for estimators_specs in estimators_specs_list:
        estimator_dir = (
            save_folder + "Trained_models/" + estimators_specs["method"] + "/" + str(estimators_specs["window_size"]) + "_points/" + str(estimators_specs["decimation_factor"]) + "_decimation_factor/" + estimators_specs["model_type"].split(".")[-1] + "/gridsearch_" + estimators_specs["gridsearch_hash"] + "/"
        )

        print("Loading -> ", estimator_dir + "best_estimator.joblib")
        estimator = jl.load(estimator_dir + "best_estimator.joblib")
        print("Loaded -> ", estimator_dir + "best_estimator.joblib")

        cache_dir = estimator_dir + "pred_cache/"
        os.makedirs(cache_dir, exist_ok=True)

        estimators_list.append(
            {
                "estimator": estimator,
                "method": estimators_specs["method"],
                "window_size": estimators_specs["window_size"],
                "decimation_factor": estimators_specs["decimation_factor"],
                "estimator_dir": estimator_dir,
                "cache_dir": cache_dir,
            }
        )

    return estimators_specs_list, estimators_list


def predict_samples(data_folder, estimators, metadata_subject):
    """
    Predict window probabilities for one subject row from metadata.
    Returns:
    - y_list: one concatenated probability vector per estimator
    - hp_tot_list: one mean(valid probability) per estimator
    - invalid_bitmap: concatenated invalid-window bitmap for the requested subjects
    """

    if not estimators:
        raise ValueError("You have selected zero estimators to predict the samples with")

    if len(set(es['window_size'] for es in estimators)) != 1:
        raise ValueError("You have selected estimators that operate on different window sizes")

    if len(set(es['decimation_factor'] for es in estimators)) != 1:
        raise ValueError("You have selected estimators that operate on different decimation factors")

    window_size = estimators[0]["window_size"]
    decimation_factor = estimators[0]["decimation_factor"]

    y_per_estimator = [None] * len(estimators)
    subject_invalid_mask = None
    subject = str(int(metadata_subject["subject"]))
    subject_df = metadata_subject.to_frame().T

    # Phase 1: per estimator, either load cached probabilities or submit an async Ray prediction task.
    # `pending_by_idx[i] = (ObjectRef, cache_path)` keeps the future result aligned with estimator index.
    method_to_features = {}
    pending_by_idx = {}
    ray_started = False
    for i, es in enumerate(estimators):
        cache_dir = es.get("cache_dir")
        cache_path = os.path.join(cache_dir, f"subject_{subject}_week.csv") if cache_dir else None

        if cache_path is not None and os.path.exists(cache_path):
            cached_df = pd.read_csv(
                cache_path,
                usecols=["proba", "is_invalid"],
                engine="pyarrow",
            )
            y_per_estimator[i] = cached_df["proba"].to_numpy(dtype=float)
            if subject_invalid_mask is None:
                subject_invalid_mask = cached_df["is_invalid"].to_numpy(dtype=np.uint8).astype(bool)
        else:
            method = es["method"]
            # Build windows once per method and reuse them across all estimators of the same method.
            if method not in method_to_features:
                X, _, _, _, invalid_bitmap = create_windows(
                    data_folder=data_folder,
                    metadata=subject_df,
                    operation_type=method,
                    WINDOW_SIZE=window_size,
                    decimation_factor=decimation_factor,
                    input_type="week",
                )
                method_to_features[method] = X
                if subject_invalid_mask is None:
                    subject_invalid_mask = np.asarray(invalid_bitmap, dtype=bool)

            if not ray_started:
                ray.init(include_dashboard=False)
                ray_started = True

            # Match train_select_classifiers-style scheduling:
            # - GPU available: 1 GPU per task and a CPU share of total_cpus / total_gpus.
            # - No GPU: each task reserves all CPUs, which enforces sequential execution.
            pending_by_idx[i] = (
                _predict_proba_remote.options(
                    num_gpus=1 if torch.cuda.device_count() > 0 else 0,
                    num_cpus=(
                        max(1, (os.cpu_count() or 1) // torch.cuda.device_count())
                        if torch.cuda.device_count() > 0
                        else (os.cpu_count() or 1)
                    ),
                ).remote(es["estimator"], method_to_features[method]),
                cache_path,
            )

    # Phase 2: resolve missing predictions on demand, optionally persist cache, then compute outputs.
    y_list = []
    hp_tot_list = []

    for i in range(len(estimators)):
        probs = y_per_estimator[i]
        if probs is None:
            # Ray task was already submitted in phase 1; `ray.get` waits only for this estimator if needed.
            ref, cache_path = pending_by_idx[i]
            probs = ray.get(ref)
            y_per_estimator[i] = probs

            if cache_path is not None:
                pd.DataFrame(
                    {
                        "window_idx": np.arange(probs.size, dtype=np.int32),
                        "proba": probs,
                        "is_invalid": subject_invalid_mask.astype(np.uint8),
                    }
                ).to_csv(cache_path, index=False)

        y_list.append(probs)
        valid_probs = probs[~subject_invalid_mask] if subject_invalid_mask is not None else np.array([], dtype=float)
        if valid_probs.size > 0:
            hp_tot_list.append(float(valid_probs.mean()))
        else:
            hp_tot_list.append(np.nan)

    if ray_started:
        ray.shutdown()

    invalid_bitmap = subject_invalid_mask.astype(np.uint8) if subject_invalid_mask is not None else np.array([], dtype=np.uint8)

    return y_list, hp_tot_list, invalid_bitmap

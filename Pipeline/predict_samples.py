import numpy as np
import joblib as jl
import os
import pandas as pd
from create_windows import create_windows


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


def predict_samples(data_folder, estimators, subject_index):
    """
    Predict window probabilities for the requested metadata indexes.
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

    # subject_indexes are metadata row indexes; we map them to subject IDs for cache filenames.
    metadata_subjects = pd.read_excel(data_folder + "metadata2023_08.xlsx", usecols=["subject"])

    # We accumulate per subject and concatenate only once at the end.
    y_chunks_per_estimator = [[] for _ in estimators]
    valid_sum = np.zeros(len(estimators), dtype=float)
    valid_count = np.zeros(len(estimators), dtype=int)
    invalid_chunks = []


    subject = metadata_subjects["subject"].iloc[int(subject_index)]

    cache_paths = []
    missing_methods = set()

    for es in estimators:
        cache_dir = es.get("cache_dir")
        cache_path = None
        if cache_dir:
            # Cache key by (estimator folder, subject id, week input type).
            cache_path = os.path.join(cache_dir, f"subject_{subject}_week.csv")
        cache_paths.append(cache_path)

        # We only build windows for methods that have at least one cache miss.
        if cache_path is None or not os.path.exists(cache_path):
            missing_methods.add(es["method"])

    method_to_features = {}
    method_invalid_mask = {}
    subject_invalid_mask = None

    for method in missing_methods:
        X, _, _, _, invalid_bitmap = create_windows(
            data_folder=data_folder,
            subjects_indexes=[int(subject_index)],
            operation_type=method,
            WINDOW_SIZE=window_size,
            decimation_factor=decimation_factor,
            input_type="week",
        )
        method_to_features[method] = X
        method_invalid_mask[method] = np.asarray(invalid_bitmap, dtype=bool)

    for i, es in enumerate(estimators):
        cache_path = cache_paths[i]

        if cache_path is not None and os.path.exists(cache_path):
            # Fast path: load only the columns needed downstream.
            cached_df = pd.read_csv(
                cache_path,
                usecols=["proba", "is_invalid"],
                engine="pyarrow",
            )
            probs = cached_df["proba"].to_numpy(dtype=float)
            invalid_mask = cached_df["is_invalid"].to_numpy(dtype=np.uint8).astype(bool)
        else:
            # Slow path: compute probabilities and persist them for next runs.
            X = method_to_features[es["method"]]
            invalid_mask = method_invalid_mask[es["method"]]

            probs = es["estimator"].predict_proba(X)[:, 1]

            if cache_path is not None:
                pd.DataFrame(
                    {
                        "window_idx": np.arange(probs.size, dtype=np.int32),
                        "proba": probs,
                        "is_invalid": invalid_mask.astype(np.uint8),
                    }
                ).to_csv(cache_path, index=False)

        # Keep raw probabilities untouched; validity is handled via invalid_mask.
        valid_probs = probs[~invalid_mask]
        valid_sum[i] += float(valid_probs.sum())
        valid_count[i] += int(valid_probs.size)
        y_chunks_per_estimator[i].append(probs)
        if subject_invalid_mask is None:
            subject_invalid_mask = invalid_mask

    invalid_chunks.append(subject_invalid_mask.astype(np.uint8))

    y_list = []
    hp_tot_list = []

    for i in range(len(estimators)):
        if y_chunks_per_estimator[i]:
            y_list.append(np.concatenate(y_chunks_per_estimator[i]))
        else:
            y_list.append(np.array([], dtype=float))

        # Mean probability across all valid windows from all requested subjects.
        if valid_count[i] > 0:
            hp_tot_list.append(float(valid_sum[i] / valid_count[i]))
        else:
            hp_tot_list.append(np.nan)

    invalid_bitmap = np.concatenate(invalid_chunks) if invalid_chunks else np.array([], dtype=np.uint8)

    return y_list, hp_tot_list, invalid_bitmap

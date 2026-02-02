# Train and compare multiple sequence classifiers across:
# - windowing strategies (method)
# - window sizes (number of points)
# - decimation factors
# - model families (LSTM/GRU/RNN/CNN1D/Transformer/Reservoir)
#
# The function optionally runs a Ray Tune job to train missing runs, then aggregates
# the stored GridSearchCV results into summary CSVs.
import json
import os
import hashlib
import pandas as pd

import random
import torch
from torch import nn
from skorch import NeuralNetClassifier
from skorch_models import (
    Conv1DSequenceClassifier,
    GRUSequenceClassifier,
    LSTMSequenceClassifier,
    RNNSequenceClassifier,
    ReservoirSequenceClassifier,
    TransformerSequenceClassifier,
)


def _safe_model_name(name: str) -> str:
    # Make a filesystem-friendly model name for use in output folder paths.
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in name)


def _hash_param_grid(param_grid: dict) -> str:
    # Create a stable short hash for a parameter grid so different grids get different output folders.
    payload = json.dumps(param_grid, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:10]


def train_select_classifiers(
    data_folder,
    save_folder,
    subjects_indexes,
    l_window_size=[300, 600, 900],
    l_method=["concat", "difference", "ai"],
    l_decimation_factor=[3],
    gridsearch_specs_list=None,
):
    # Main orchestration entry point:
    # 1) define (or accept) model specs + parameter grids,
    # 2) enumerate all combinations of method/window/model/decimation,
    # 3) train missing combinations (distributed via Ray),
    # 4) load and aggregate cv_results.csv for all runs.

    if gridsearch_specs_list is None:
        # Default model specs used when the caller doesn't provide their own list.
        # NOTE: device selection is passed into skorch (not into the PyTorch modules directly).
        device = "cuda" if torch.cuda.is_available() else "cpu"
        skorch_common = {
            "criterion": nn.CrossEntropyLoss,
            "max_epochs": 200,
            "lr": 1e-3,
            "batch_size": 64,
            "iterator_train__shuffle": True,
            "train_split": None,
            "device": device,
            "verbose": 0,
        }

        gridsearch_specs_list = [
            {
                "name": "LSTM",
                # skorch wrapper around a local PyTorch module (LSTMSequenceClassifier).
                "estimator": NeuralNetClassifier(module=LSTMSequenceClassifier, **skorch_common),
                "param_grid": {
                    "lr": [1e-3],
                    "module__hidden_size": [16],
                },
            },
            {
                "name": "GRU",
                "estimator": NeuralNetClassifier(module=GRUSequenceClassifier, **skorch_common),
                "param_grid": {
                    "lr": [1e-3],
                    "module__hidden_size": [16],
                },
            },
            {
                "name": "RNN",
                "estimator": NeuralNetClassifier(module=RNNSequenceClassifier, **skorch_common),
                "param_grid": {
                    "lr": [1e-3],
                    "module__hidden_size": [16],
                },
            },
            {
                "name": "CNN1D",
                "estimator": NeuralNetClassifier(module=Conv1DSequenceClassifier, **skorch_common),
                "param_grid": {
                    "lr": [1e-3],
                    "module__channels": [16],
                    "module__kernel_size": [5],
                },
            },
            {
                "name": "Transformer",
                "estimator": NeuralNetClassifier(
                    module=TransformerSequenceClassifier,
                    # Transformers can be heavier; use shorter training by default here.
                    **{**skorch_common, "max_epochs": 3, "batch_size": 32},
                ),
                "param_grid": {
                    "lr": [1e-3],
                    "module__d_model": [32],
                    "module__nhead": [4],
                    "module__num_layers": [2],
                    "module__patch_size": [32],
                },
            },
            {
                "name": "Reservoir",
                "estimator": NeuralNetClassifier(
                    module=ReservoirSequenceClassifier,
                    # Reservoir-based models are also kept short by default.
                    **{**skorch_common, "max_epochs": 3, "batch_size": 64},
                ),
                "param_grid": {
                    "lr": [1e-3],
                    "module__reservoir_size": [200],
                    "module__downsample": [16],
                },
            },
        ]

    # Aggregation happens after Tune completes by iterating over Tune's trial configs and reading each run's
    # `cv_results.csv` from its deterministic output folder.

    # Capture the directory containing this file so Ray workers can reliably `chdir` for local imports.
    pipeline_dir = os.path.dirname(os.path.abspath(__file__))
    if hasattr(subjects_indexes, "tolist"):
        # Normalize pandas/numpy index-like objects to a plain Python list for serialization and Ray.
        subjects_indexes = subjects_indexes.tolist()

    def _artifacts_exist(gridsearch_folder: str) -> bool:
        # Key artifacts produced by train_best_model.
        return os.path.exists(gridsearch_folder + "best_estimator.joblib") and os.path.exists(
            gridsearch_folder + "GridSearchCV_stats/cv_results.csv"
        )

    def _gridsearch_folder_for_task(method: str, window_size: int, decimation_factor: int, gridsearch_specs: dict) -> str:
        # Folder layout encodes the experimental choices; hash disambiguates different param grids.
        model_name = _safe_model_name(gridsearch_specs["name"])
        gridsearch_hash = _hash_param_grid(gridsearch_specs["param_grid"])
        return f"{save_folder}Trained_models/{method}/{window_size}_points/{decimation_factor}_decimation_factor/{model_name}/gridsearch_{gridsearch_hash}/"

    # Train the full Cartesian grid via Ray Tune. Each trial will early-exit if artifacts already exist.
    # Ray prints warnings/errors related to accelerator env var overrides and metrics exporting.
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

    import ray
    from ray import tune
    from ray.tune import RunConfig

    spec_indices = list(range(len(gridsearch_specs_list)))

    def _ray_train_best_model(
        config,
        pipeline_dir,
        data_folder,
        subjects_indexes,
        gridsearch_specs_list,
    ):
        # Ray Tune trainable: receives one point in the Cartesian product.
        method = config["method"]
        window_size = int(config["window_size"])
        decimation_factor = int(config["decimation_factor"])
        spec_idx = int(config["spec_idx"])

        gridsearch_specs = gridsearch_specs_list[spec_idx]
        gridsearch_folder = _gridsearch_folder_for_task(method, window_size, decimation_factor, gridsearch_specs)

        # Skip work if the run was already trained.
        if _artifacts_exist(gridsearch_folder):
            return

        # Ensure local imports resolve consistently inside the Ray worker process.
        os.chdir(pipeline_dir)

        # Fixed seed for reproducibility across workers.
        seed = 42
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Import inside the Ray worker after chdir so local imports work.
        from train_best_model import train_best_model

        train_best_model(
            data_folder,
            subjects_indexes,
            gridsearch_folder,
            gridsearch_specs["estimator"],
            gridsearch_specs["param_grid"],
            method,
            window_size,
            decimation_factor,
        )

    # Start (or connect to) a Ray runtime. When GPUs are available, expose them to Ray for scheduling.
    ray.init(
        ignore_reinit_error=True,
        log_to_driver=True,
        num_gpus=(torch.cuda.device_count() if torch.cuda.is_available() else 0),
    )

    # Allocate one GPU per trial when available; otherwise allocate one CPU per trial for parallelism.
    resources = {"gpu": 1, "cpu": 1} if torch.cuda.is_available() else {"cpu": 1}

    # Bind constant parameters once; Ray will vary only the grid_search dimensions in `param_space`.
    trainable = tune.with_parameters(
        _ray_train_best_model,
        pipeline_dir=pipeline_dir,
        data_folder=data_folder,
        subjects_indexes=subjects_indexes,
        gridsearch_specs_list=gridsearch_specs_list,
    )

    param_space = {
        "method": tune.grid_search(l_method),
        "window_size": tune.grid_search(l_window_size),
        "decimation_factor": tune.grid_search(l_decimation_factor),
        "spec_idx": tune.grid_search(spec_indices),
    }
    tuner = tune.Tuner(
        tune.with_resources(trainable, resources),
        # One Ray trial per Cartesian point (some will early-exit if already trained).
        param_space=param_space,
        run_config=RunConfig(name="train_select_classifiers", verbose=1),
    )
    result_grid = tuner.fit()
    ray.shutdown()

    estimators_l = []
    best_estimators_l = []
    for result in result_grid:
        config = getattr(result, "config", None) or {}
        try:
            method = config["method"]
            window_size = int(config["window_size"])
            decimation_factor = int(config["decimation_factor"])
            spec_idx = int(config["spec_idx"])
        except (KeyError, TypeError, ValueError):
            continue

        if spec_idx < 0 or spec_idx >= len(gridsearch_specs_list):
            continue

        gridsearch_specs = gridsearch_specs_list[spec_idx]
        gridsearch_folder = _gridsearch_folder_for_task(method, window_size, decimation_factor, gridsearch_specs)
        cv_results_path = gridsearch_folder + "GridSearchCV_stats/cv_results.csv"
        if not os.path.exists(cv_results_path):
            continue

        cv_results = pd.read_csv(cv_results_path, index_col=0)
        cv_results.columns = cv_results.columns.str.strip()

        model_name = _safe_model_name(gridsearch_specs["name"])
        gridsearch_hash = _hash_param_grid(gridsearch_specs["param_grid"])
        cv_results = cv_results.assign(
            method=method,
            window_size=window_size,
            decimation_factor=decimation_factor,
            model_type=model_name,
            gridsearch_hash=gridsearch_hash,
        )
        estimators_l.append(cv_results)
        best_estimators_l.append(cv_results.iloc[[cv_results["rank_test_score"].argmin()]])

    if not estimators_l:
        # No results were found/loaded (e.g., training failed or folder layout mismatch).
        print("No compatible estimators were trained for the selected methods/window sizes.")
        return

    # Concatenate all runs into two summary tables: all candidates, and the best candidate per run.
    estimators_df = pd.concat(estimators_l, ignore_index=True)
    best_estimators_df = pd.concat(best_estimators_l, ignore_index=True)

    # Sort so top-performing models appear first, then write summaries to the save folder.
    estimators_df.sort_values(by=['mean_test_score', 'std_test_score'], ascending=False).to_csv(save_folder+'estimators_results.csv')
    best_estimators_df.sort_values(by=['mean_test_score', 'std_test_score'], ascending=False).to_csv(save_folder+'best_estimators_results.csv')

# Train and compare multiple sequence classifiers across:
# - windowing strategies (method)
# - window sizes (number of points)
# - decimation factors
# - model families (LSTM/GRU/RNN/CNN1D/Transformer/Reservoir)
#
# The function runs a Ray Tune job over the full Cartesian grid; each trial skips work
# when its artifacts are already present on disk, then cv_results.csv files are aggregated
# into summary CSVs.
import json
import os
import hashlib
import pandas as pd

import random
import torch
from torch import nn
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
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
    # 2) define a Cartesian grid for Ray Tune,
    # 3) train missing combinations (trials early-exit when artifacts already exist),
    # 4) load and aggregate cv_results.csv across the selected grid.

    if gridsearch_specs_list is None:
        # Default model specs used when the caller doesn't provide their own list.
        # NOTE: device selection is passed into skorch (not into the PyTorch modules directly).
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def _default_callbacks():
            # Create fresh callback objects per estimator to avoid shared state across clones/fits.
            return [
                (
                    "early_stopping",
                    EarlyStopping(
                        monitor="valid_loss",
                        patience=15,
                        threshold=1e-4,
                        threshold_mode="rel",
                        lower_is_better=True,
                        load_best=True,
                    ),
                )
            ]

        skorch_common = {
            "criterion": nn.CrossEntropyLoss,
            "max_epochs": 100,
            "lr": 1e-3,
            "batch_size": 64,
            "optimizer": torch.optim.AdamW,
            "iterator_train__shuffle": True,
            # Keep a small validation split inside each CV fold so we can use early stopping.
            "train_split": ValidSplit(0.2, stratified=True, random_state=42),
            "device": device,
            "verbose": 0,
        }

        gridsearch_specs_list = [
            {
                "name": "LSTM",
                "estimator": NeuralNetClassifier(
                    module=LSTMSequenceClassifier,
                    callbacks=_default_callbacks(),
                    **skorch_common,
                ),
                "param_grid": {
                    "optimizer__weight_decay": [0.0, 1e-4],
                    "lr": [3e-4, 1e-3],
                    "module__hidden_size": [32, 64],
                    "module__num_layers": [1, 2],
                    "module__dropout": [0.0, 0.2],
                    "module__bidirectional": [False, True],
                },
            },
            {
                "name": "GRU",
                "estimator": NeuralNetClassifier(
                    module=GRUSequenceClassifier,
                    callbacks=_default_callbacks(),
                    **skorch_common,
                ),
                "param_grid": {
                    "optimizer__weight_decay": [0.0, 1e-4],
                    "lr": [3e-4, 1e-3],
                    "module__hidden_size": [32, 64],
                    "module__num_layers": [1, 2],
                    "module__dropout": [0.0, 0.2],
                    "module__bidirectional": [False, True],
                },
            },
            {
                "name": "RNN",
                "estimator": NeuralNetClassifier(
                    module=RNNSequenceClassifier,
                    callbacks=_default_callbacks(),
                    **skorch_common,
                ),
                "param_grid": {
                    "optimizer__weight_decay": [0.0, 1e-4],
                    "lr": [3e-4, 1e-3],
                    "module__hidden_size": [32, 64],
                    "module__num_layers": [1, 2],
                    "module__dropout": [0.0, 0.2],
                    "module__bidirectional": [False],
                    "module__nonlinearity": ["tanh", "relu"],
                },
            },
            {
                "name": "CNN1D",
                "estimator": NeuralNetClassifier(
                    module=Conv1DSequenceClassifier,
                    callbacks=_default_callbacks(),
                    **skorch_common,
                ),
                "param_grid": {
                    "optimizer__weight_decay": [0.0, 1e-4],
                    "lr": [3e-4, 1e-3],
                    "module__channels": [16, 32, 64],
                    "module__kernel_size": [5, 7],
                    "module__dropout": [0.0, 0.3],
                },
            },
            {
                "name": "Transformer",
                "estimator": NeuralNetClassifier(
                    module=TransformerSequenceClassifier,
                    callbacks=_default_callbacks(),
                    **skorch_common,
                ),
                # Use a list of grids to avoid invalid (d_model, nhead) combinations.
                "param_grid": [
                    {
                        "optimizer__weight_decay": [1e-4],
                        "lr": [3e-4, 1e-3],
                        "batch_size": [16, 32],
                        "module__d_model": [32],
                        "module__nhead": [4, 8],
                        "module__num_layers": [2, 3],
                        "module__dim_feedforward": [128],
                        "module__dropout": [0.1],
                        "module__patch_size": [32],
                    },
                    {
                        "optimizer__weight_decay": [1e-4],
                        "lr": [3e-4, 1e-3],
                        "batch_size": [16, 32],
                        "module__d_model": [64],
                        "module__nhead": [8],
                        "module__num_layers": [2, 3],
                        "module__dim_feedforward": [256],
                        "module__dropout": [0.1],
                        "module__patch_size": [32],
                    },
                ],
            },
            {
                "name": "Reservoir",
                "estimator": NeuralNetClassifier(
                    module=ReservoirSequenceClassifier,
                    callbacks=_default_callbacks(),
                    **skorch_common,
                ),
                "param_grid": {
                    "optimizer__weight_decay": [0.0],
                    "lr": [1e-3],
                    "module__reservoir_size": [200, 400],
                    "module__spectral_radius": [0.8, 0.9, 1.0],
                    "module__leak_rate": [1.0],
                    "module__input_scaling": [0.2, 0.5],
                    "module__downsample": [8, 16],
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

    model_names = [spec["name"] for spec in gridsearch_specs_list]
    if len(set(model_names)) != len(model_names):
        raise ValueError(
            "Each entry in gridsearch_specs_list must have a unique 'name' so Ray trials can be labeled clearly."
        )
    model_name_to_idx = {name: idx for idx, name in enumerate(model_names)}

    def _ray_train_best_model(
        config,
        pipeline_dir,
        data_folder,
        subjects_indexes,
        gridsearch_specs_list,
        model_name_to_idx,
    ):
        # Ray Tune trainable: receives one point in the Cartesian product.
        method = config["method"]
        window_size = int(config["window_size"])
        decimation_factor = int(config["decimation_factor"])
        model_name = config["model_name"]
        spec_idx = model_name_to_idx[model_name]

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

    # Start (or connect to) a Ray runtime.
    ray.init(include_dashboard=False)

    # Allocate one GPU per trial when available; otherwise allocate all CPU cores to a single trial.
    resources = {"gpu": 1, "cpu": max(1, (os.cpu_count() or 1) // max(1, torch.cuda.device_count()))} if torch.cuda.is_available() else {"cpu": os.cpu_count() or 1}

    # Bind constant parameters once; Ray will vary only the grid_search dimensions in `param_space`.
    trainable = tune.with_parameters(
        _ray_train_best_model,
        pipeline_dir=pipeline_dir,
        data_folder=data_folder,
        subjects_indexes=subjects_indexes,
        gridsearch_specs_list=gridsearch_specs_list,
        model_name_to_idx=model_name_to_idx,
    )

    param_space = {
        "method": tune.grid_search(l_method),
        "window_size": tune.grid_search(l_window_size),
        "decimation_factor": tune.grid_search(l_decimation_factor),
        "model_name": tune.grid_search(model_names),
    }

    def _trial_name_creator(trial) -> str:
        config = getattr(trial, "config", {}) or {}
        model_name = _safe_model_name(str(config.get("model_name", "model")))
        trial_id = str(getattr(trial, "trial_id", "trial"))
        return f"{model_name}_{trial_id}"

    def _trial_dirname_creator(trial) -> str:
        return _safe_model_name(_trial_name_creator(trial))

    tuner_kwargs = {
        # One Ray trial per Cartesian point (some will early-exit if already trained).
        "param_space": param_space,
        "run_config": RunConfig(name="train_select_classifiers", verbose=1),
    }
    TuneConfig = getattr(tune, "TuneConfig", None)
    if TuneConfig is not None:
        try:
            tuner_kwargs["tune_config"] = TuneConfig(
                trial_name_creator=_trial_name_creator,
                trial_dirname_creator=_trial_dirname_creator,
            )
        except TypeError:
            tuner_kwargs["tune_config"] = TuneConfig(trial_name_creator=_trial_name_creator)

    tuner = tune.Tuner(
        tune.with_resources(trainable, resources),
        **tuner_kwargs,
    )
    result_grid = tuner.fit()
    ray.shutdown()

    estimators_l = []
    best_estimators_l = []
    for result in result_grid:
        # `result_grid` contains one entry per Tune trial (i.e., one point in the Cartesian product).
        # We use the trial `config` to deterministically reconstruct the output folder where
        # train_best_model wrote the GridSearchCV statistics, then load and tag cv_results.csv.
        config = result.config
        method = config["method"]
        window_size = int(config["window_size"])
        decimation_factor = int(config["decimation_factor"])
        model_name = config["model_name"]
        spec_idx = model_name_to_idx[model_name]

        gridsearch_specs = gridsearch_specs_list[spec_idx]
        gridsearch_folder = _gridsearch_folder_for_task(method, window_size, decimation_factor, gridsearch_specs)
        cv_results_path = gridsearch_folder + "GridSearchCV_stats/cv_results.csv"
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

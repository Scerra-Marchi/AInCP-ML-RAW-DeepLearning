import json
import os
import hashlib
import itertools
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

#import warnings 

#warnings.filterwarnings("ignore")

def _safe_model_name(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in name)


def _hash_param_grid(param_grid: dict) -> str:
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

    if gridsearch_specs_list is None:
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
                    **{**skorch_common, "max_epochs": 3, "batch_size": 64},
                ),
                "param_grid": {
                    "lr": [1e-3],
                    "module__reservoir_size": [200],
                    "module__downsample": [16],
                },
            },
        ]

    estimators_l = []
    best_estimators_l = []

    pipeline_dir = os.path.dirname(os.path.abspath(__file__))
    if hasattr(subjects_indexes, "tolist"):
        subjects_indexes = subjects_indexes.tolist()

    runs = []
    train_tasks = []

    indexed_specs = list(enumerate(gridsearch_specs_list))

    for method, window_size, (spec_idx, gridsearch_specs), decimation_factor in itertools.product(
        l_method, l_window_size, indexed_specs, l_decimation_factor
    ):

        model_name = _safe_model_name(gridsearch_specs["name"])
        param_grid = gridsearch_specs["param_grid"]

        gridsearch_hash = _hash_param_grid(param_grid)

        gridsearch_folder = f"{save_folder}Trained_models/{method}/{window_size}_points/{decimation_factor}_decimation_factor/{model_name}/gridsearch_{gridsearch_hash}/"

        run = {
            "spec_idx": spec_idx,
            "method": method,
            "window_size": window_size,
            "decimation_factor": decimation_factor,
            "model_name": model_name,
            "gridsearch_hash": gridsearch_hash,
            "gridsearch_folder": gridsearch_folder,
            "param_grid": param_grid,
        }
        runs.append(run)

        if not (os.path.exists(gridsearch_folder + "best_estimator.joblib")) or not (
            os.path.exists(gridsearch_folder + "GridSearchCV_stats/cv_results.csv")
        ):
            train_tasks.append(run)

    if train_tasks:
        import ray
        from ray import tune
        from ray.tune import RunConfig

        def _ray_train_best_model(config, pipeline_dir, data_folder, subjects_indexes, gridsearch_specs_list):
            task = config["task"]

            os.chdir(pipeline_dir)

            seed = 42
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # Import inside the Ray worker after chdir so local imports work.
            from train_best_model import train_best_model

            gridsearch_specs = gridsearch_specs_list[int(task["spec_idx"])]

            train_best_model(
                data_folder,
                subjects_indexes,
                task["gridsearch_folder"],
                gridsearch_specs["estimator"],
                gridsearch_specs["param_grid"],
                task["method"],
                task["window_size"],
                task["decimation_factor"],
            )

        ray.init(ignore_reinit_error=True, log_to_driver=True)

        resources = {"gpu": 1} if torch.cuda.is_available() else {"cpu": (os.cpu_count() or 1)}

        trainable = tune.with_parameters(
            _ray_train_best_model,
            pipeline_dir=pipeline_dir,
            data_folder=data_folder,
            subjects_indexes=subjects_indexes,
            gridsearch_specs_list=gridsearch_specs_list,
        )
        tuner = tune.Tuner(
            tune.with_resources(trainable, resources),
            param_space={"task": tune.grid_search(train_tasks)},
            run_config=RunConfig(name="train_select_classifiers", verbose=1),
        )
        tuner.fit()

    for run in runs:
        cv_results = pd.read_csv(
            run["gridsearch_folder"] + "GridSearchCV_stats/cv_results.csv",
            index_col=0,
        )
        cv_results.columns = cv_results.columns.str.strip()
        cv_results = cv_results.assign(
            method=run["method"],
            window_size=run["window_size"],
            decimation_factor=run["decimation_factor"],
            model_type=run["model_name"],
            gridsearch_hash=run["gridsearch_hash"],
        )

        estimators_l.append(cv_results)
        best_estimators_l.append(cv_results.iloc[[cv_results['rank_test_score'].argmin()]])

    if not estimators_l:
        print("No compatible estimators were trained for the selected methods/window sizes.")
        return

    estimators_df = pd.concat(estimators_l, ignore_index=True)
    best_estimators_df = pd.concat(best_estimators_l, ignore_index=True)

    estimators_df.sort_values(by=['mean_test_score', 'std_test_score'], ascending=False).to_csv(save_folder+'estimators_results.csv')
    best_estimators_df.sort_values(by=['mean_test_score', 'std_test_score'], ascending=False).to_csv(save_folder+'best_estimators_results.csv')

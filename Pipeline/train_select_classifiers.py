import json
import os
import hashlib
import itertools
import pandas as pd
from train_best_model import train_best_model
from estimator_io import BEST_ESTIMATOR_FILENAME

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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
        gridsearch_specs_list = [
            {
                "name": "KMeans",
                "estimator": KMeans(n_clusters=2, random_state=42, n_init="auto"),
                "param_grid": {"init": ["k-means++", "random"]},
            },
            {
                "name": "LogisticRegression",
                "estimator": LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"),
                "param_grid": {"C": [0.1, 1.0, 10.0]},
            },
            {
                "name": "RandomForestClassifier",
                "estimator": RandomForestClassifier(random_state=42, class_weight="balanced"),
                "param_grid": {"n_estimators": [200, 500], "max_depth": [None, 8, 16]},
            },
        ]
        # To try a PyTorch LSTM with sklearn GridSearchCV (via skorch), pass an explicit
        # `gridsearch_specs_list` from your entrypoint and include something like:
        #   from skorch import NeuralNetClassifier
        #   from torch import nn
        #   from torch_skorch import LSTMSequenceClassifier
        #   lstm = NeuralNetClassifier(
        #       module=LSTMSequenceClassifier,
        #       criterion=nn.CrossEntropyLoss,
        #       max_epochs=20,
        #       lr=1e-3,
        #       batch_size=64,
        #       iterator_train__shuffle=True,
        #       device="cpu",
        #   )
        #   {"name": "LSTM", "estimator": lstm, "param_grid": {"lr": [1e-3], "module__hidden_size": [64, 128]}}

    estimators_l = []
    best_estimators_l = []

    for method, window_size, gridsearch_specs, decimation_factor in itertools.product(
        l_method, l_window_size, gridsearch_specs_list, l_decimation_factor
    ):

        model_name = _safe_model_name(gridsearch_specs["name"])
        estimator = gridsearch_specs["estimator"]
        param_grid = gridsearch_specs["param_grid"]

        gridsearch_hash = _hash_param_grid(param_grid)

        print(
            "Method: ",
            method,
            "\nWindow size: ",
            window_size,
            "\nModel type: ",
            model_name,
            "\nDecimation factor: ",
            decimation_factor,
            "\nGrid search params: ",
            param_grid,
        )

        gridsearch_folder = (
            save_folder
            + "Trained_models/"
            + method
            + "/"
            + str(window_size)
            + "_points/"
            + str(decimation_factor)
            + "_decimation_factor/"
            + model_name
            + "/"
            + "gridsearch_"
            + gridsearch_hash
            + "/"
        )

        if not (os.path.exists(gridsearch_folder + BEST_ESTIMATOR_FILENAME)) or not (
            os.path.exists(gridsearch_folder + "GridSearchCV_stats/cv_results.csv")
        ):

            train_best_model(
                data_folder,
                subjects_indexes,
                gridsearch_folder,
                estimator,
                param_grid,
                method,
                window_size,
                decimation_factor,
            )

        cv_results = pd.read_csv(gridsearch_folder + 'GridSearchCV_stats/cv_results.csv', index_col=0)
        cv_results.columns = cv_results.columns.str.strip()
        cv_results['method'] = method
        cv_results['window_size'] = window_size
        cv_results['decimation_factor'] = decimation_factor
        cv_results['model_type'] = model_name
        cv_results['gridsearch_hash'] = gridsearch_hash

        estimators_l.append(cv_results)
        best_estimators_l.append(cv_results.iloc[[cv_results['rank_test_score'].argmin()]])

    estimators_df = pd.concat(estimators_l, ignore_index=True)
    best_estimators_df = pd.concat(best_estimators_l, ignore_index=True)

    estimators_df.sort_values(by=['mean_test_score', 'std_test_score'], ascending=False).to_csv(save_folder+'estimators_results.csv')
    best_estimators_df.sort_values(by=['mean_test_score', 'std_test_score'], ascending=False).to_csv(save_folder+'best_estimators_results.csv')
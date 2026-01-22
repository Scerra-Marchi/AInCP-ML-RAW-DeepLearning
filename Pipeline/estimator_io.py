import os
from typing import Any

import joblib


BEST_ESTIMATOR_FILENAME = "best_estimator.joblib"
BEST_ESTIMATOR_STATS_FILENAME = "best_estimator_stats.json"

def save_best_estimator(gridsearch_folder: str, estimator: Any) -> None:
    os.makedirs(gridsearch_folder, exist_ok=True)
    joblib.dump(estimator, os.path.join(gridsearch_folder, BEST_ESTIMATOR_FILENAME))
    # Stats (including hemi_cluster) are written by the training pipeline in
    # `GridSearchCV_stats/best_estimator_stats.json`.


def load_best_estimator(gridsearch_folder: str) -> tuple[Any, int]:
    estimator_path = os.path.join(gridsearch_folder, BEST_ESTIMATOR_FILENAME)
    stats_path = os.path.join(gridsearch_folder, "GridSearchCV_stats", BEST_ESTIMATOR_STATS_FILENAME)

    estimator = joblib.load(estimator_path)
    hemi_cluster = 1
    if os.path.exists(stats_path):
        import json

        with open(stats_path, "r") as file:
            stats = json.load(file)
        hemi_cluster = int(stats.get("Hemi cluster", 1))

    return estimator, hemi_cluster

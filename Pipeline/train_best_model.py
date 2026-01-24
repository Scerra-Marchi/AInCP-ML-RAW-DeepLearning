import os
import json
import pandas as pd
import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from create_windows import create_windows
import sys
from skorch_models import save_best_estimator

import warnings
warnings.filterwarnings("ignore")   #TODO: remove this line when the code is stable

def _best_hemi_cluster_from_predictions(y_true: np.ndarray, y_pred_labels: np.ndarray) -> int:
    labels = np.unique(y_pred_labels)
    if labels.size != 2:
        raise ValueError(f"Expected 2 clusters/classes, got labels={labels.tolist()}")

    best_label = int(labels[0])
    best_score = -1.0
    for candidate in labels:
        y_pred_bin = (y_pred_labels == candidate).astype(int)
        score = f1_score(y_true, y_pred_bin, average="weighted")
        if score > best_score:
            best_score = float(score)
            best_label = int(candidate)
    return best_label


def scorer_f(estimator, X, y):
    y_pred = estimator.predict(X)
    if is_classifier(estimator):
        return f1_score(y, y_pred, average="weighted")
    hemi_cluster = _best_hemi_cluster_from_predictions(np.asarray(y), np.asarray(y_pred))
    y_pred_bin = (np.asarray(y_pred) == hemi_cluster).astype(int)
    return f1_score(y, y_pred_bin, average="weighted")

def _is_skorch_estimator(estimator) -> bool:
    return estimator.__class__.__module__.startswith("skorch.")


def train_best_model(data_folder, subjects_indexes, gridsearch_folder, estimator, param_grid, method, window_size, decimation_factor) -> bool:
    model = clone(estimator)

    X, _, _, y = create_windows(data_folder, subjects_indexes, method, window_size, decimation_factor)
    X = np.asarray(X)
    y = np.asarray(y)

    is_skorch = _is_skorch_estimator(model)

    effective_param_grid = dict(param_grid)
    if is_skorch:
        n_features = int(X.shape[-1]) if X.ndim == 3 else 1
        model_params = model.get_params(deep=True)
        if "module__input_size" in model_params and "module__input_size" not in effective_param_grid:
            effective_param_grid["module__input_size"] = [n_features]
        if "module__in_channels" in model_params and "module__in_channels" not in effective_param_grid:
            effective_param_grid["module__in_channels"] = [n_features]
 
    #                                                             dobbiamo fixare il seed? FATTP
    n_jobs = 1 if is_skorch else -1
    parameter_tuning_method = GridSearchCV(
        model,
        effective_param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=n_jobs,
        return_train_score=True,
        verbose=3,
        scoring=scorer_f,
    )
    parameter_tuning_method.fit(X, y)

    estimator = parameter_tuning_method.best_estimator_

    hemi_cluster = 1
    if not is_classifier(estimator):
        y_pred_labels = estimator.predict(X)
        hemi_cluster = _best_hemi_cluster_from_predictions(np.asarray(y), np.asarray(y_pred_labels))

    # print('y = ', y)
    # print('y_pred = ', y_pred)
    # print('f1_score = ', f1_score(y, y_pred, average='weighted'))
    # print('f1_score (inverted) = ', f1_score(y, inverted_y_pred, average='weighted'))
    # print('hemi_cluster = ', hemi_cluster, ' (1 = non invertito)')

    stats_folder = gridsearch_folder + 'GridSearchCV_stats/'
    os.makedirs(stats_folder, exist_ok = True)
    pd.DataFrame(parameter_tuning_method.cv_results_).to_csv(stats_folder + "cv_results.csv")
    
    with open(stats_folder + 'best_estimator_stats.json', 'w') as f:
        f.write(
            json.dumps(
                {
                    "Best index": int(parameter_tuning_method.best_index_),
                    "Best score": float(parameter_tuning_method.best_score_),
                    "Refit time": float(parameter_tuning_method.refit_time_),
                    "Best params": parameter_tuning_method.best_params_,
                    "Hemi cluster": int(hemi_cluster),
                },
                indent=4,
            )
        )

    save_best_estimator(gridsearch_folder, estimator)
    print('Best estimator saved\n\n------------------------------------------------\n')
    sys.stdout.flush()
    sys.stderr.flush()
    return True

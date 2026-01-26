import os
import json
import pandas as pd
import numpy as np
import joblib as jl
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from create_windows import create_windows
import sys

import warnings
warnings.filterwarnings("ignore")   #TODO: remove this line when the code is stable

def scorer_f(estimator, X, y):
    y_pred = estimator.predict(X)
    return f1_score(y, y_pred, average="weighted")


def train_best_model(data_folder, subjects_indexes, gridsearch_folder, estimator, param_grid, method, window_size, decimation_factor):
    model = clone(estimator)

    X, _, _, y = create_windows(data_folder, subjects_indexes, method, window_size, decimation_factor)
    X = np.asarray(X)
    y = np.asarray(y)

    effective_param_grid = dict(param_grid)
    n_features = int(X.shape[-1]) if X.ndim == 3 else 1
    model_params = model.get_params(deep=True)
    if "module__input_size" in model_params and "module__input_size" not in effective_param_grid:
        effective_param_grid["module__input_size"] = [n_features]
    if "module__in_channels" in model_params and "module__in_channels" not in effective_param_grid:
        effective_param_grid["module__in_channels"] = [n_features]
 
    #                                                             dobbiamo fixare il seed? FATTP
    parameter_tuning_method = GridSearchCV(
        model,
        effective_param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=1,
        return_train_score=True,
        verbose=3,
        scoring=scorer_f,
    )
    parameter_tuning_method.fit(X, y)

    estimator = parameter_tuning_method.best_estimator_

    # print('y = ', y)
    # print('y_pred = ', y_pred)
    # print('f1_score = ', f1_score(y, y_pred, average='weighted'))

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
                },
                indent=4,
            )
        )

    os.makedirs(gridsearch_folder, exist_ok=True)
    jl.dump(estimator, os.path.join(gridsearch_folder, "best_estimator.joblib"))
    print('Best estimator saved\n\n------------------------------------------------\n')
    sys.stdout.flush()
    sys.stderr.flush()

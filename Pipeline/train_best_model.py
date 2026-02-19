import os
import json
import pandas as pd
import numpy as np
import joblib as jl
import torch
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from create_windows import create_windows
from skorch_models import save_best_estimator_plots


def train_best_model(data_folder, metadata, gridsearch_folder, estimator, param_grid, method, window_size, decimation_factor):
    X, _, _, y, _ = create_windows(
        data_folder=data_folder,
        metadata=metadata,
        operation_type=method,
        WINDOW_SIZE=window_size,
        decimation_factor=decimation_factor,
        input_type='AHA',
    )
    X = np.asarray(X)
    # lo trasformiamo in float per utilizzare la BCE
    y = np.asarray(y, dtype=np.float32)
    counts = np.bincount(y.astype(np.int64), minlength=2)
    pos_weight_value = counts[0] / max(counts[1], 1)
    estimator_with_weight = clone(estimator)
    estimator_with_weight.set_params(
        criterion__pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32)
    )


    parameter_tuning_method = GridSearchCV(
        estimator_with_weight,
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=1,
        return_train_score=True,
        verbose=0,
        scoring="f1_macro", # Equal importance for classes via macro averaging.
    )
    parameter_tuning_method.fit(X, y)

    estimator = parameter_tuning_method.best_estimator_

    stats_folder = gridsearch_folder + 'GridSearchCV_stats/'
    os.makedirs(stats_folder, exist_ok = True)
    pd.DataFrame(parameter_tuning_method.cv_results_).to_csv(stats_folder + "cv_results.csv")
    save_best_estimator_plots(estimator, stats_folder, loss_label="BCEWithLogitsLoss")
    
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

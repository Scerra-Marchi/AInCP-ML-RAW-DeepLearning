import os
import json
from tempfile import TemporaryDirectory
import pandas as pd
import numpy as np
import joblib as jl
import torch
from joblib import Memory
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from create_windows import create_windows
from skorch_models import save_best_estimator_plots, set_global_determinism


def train_best_model(data_folder, metadata, gridsearch_folder, estimator, param_grid, method, window_size, decimation_factor):
    set_global_determinism()
    X, _, _, y, _, groups = create_windows(
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
    estimator_params = estimator_with_weight.get_params(deep=True)
    pos_weight_param = "model__criterion__pos_weight"
    scale_pos_weight_param = "model__scale_pos_weight"
    y_fit = y.astype(np.float32) if pos_weight_param in estimator_params else y
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    if pos_weight_param in estimator_params:
        estimator_with_weight.set_params(
            **{pos_weight_param: torch.tensor(pos_weight_value, dtype=torch.float32)}
        )
    elif scale_pos_weight_param in estimator_params:
        estimator_with_weight.set_params(**{scale_pos_weight_param: float(pos_weight_value)})

    os.makedirs(gridsearch_folder, exist_ok=True)
    with TemporaryDirectory(prefix="pipeline_cache_", dir=gridsearch_folder) as cache_dir:
        estimator_with_weight.set_params(memory=Memory(location=cache_dir, verbose=0))
        parameter_tuning_method = GridSearchCV(
            estimator_with_weight,
            param_grid,
            cv=cv.split(X, y, groups),
            n_jobs=1,
            return_train_score=True,
            verbose=0,
            refit=True,
            scoring="f1_macro", # Equal importance for classes via macro averaging.
        )
        parameter_tuning_method.fit(X, y_fit)
        estimator = parameter_tuning_method.best_estimator_
        estimator.set_params(memory=None)
        cv_results_df = pd.DataFrame(parameter_tuning_method.cv_results_)
        best_estimator_stats = {
            "Best index": int(parameter_tuning_method.best_index_),
            "Best score": float(parameter_tuning_method.best_score_),
            "Refit time": float(parameter_tuning_method.refit_time_),
            "Best params": parameter_tuning_method.best_params_,
        }

    stats_folder = gridsearch_folder + 'GridSearchCV_stats/'
    os.makedirs(stats_folder, exist_ok = True)
    cv_results_df.to_csv(stats_folder + "cv_results.csv")
    plot_estimator = estimator.named_steps["model"]
    save_best_estimator_plots(plot_estimator, stats_folder, loss_label="BCEWithLogitsLoss")
    
    with open(stats_folder + 'best_estimator_stats.json', 'w') as f:
        f.write(json.dumps(best_estimator_stats, indent=4))

    jl.dump(estimator, os.path.join(gridsearch_folder, "best_estimator.joblib"))

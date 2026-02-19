import json
import numpy as np
import pandas as pd
import joblib as jl
from sklearn.metrics import r2_score
from predict_samples import build_estimators_list, predict_samples
from train_regressor import (
    regressor_hash_from_estimators,
    REGRESSOR_PARAM_GRID,
    build_regressor_sequence,
    stack_regressor_sequences,
)

def test_classifier_regressor(
    data_folder,
    save_folder,
    metadata,
    min_mean_test_score=None,
    window_size=None,
    decimation_factor=None,
):
    best_estimators_df = pd.read_csv(save_folder + 'best_estimators_results.csv', index_col=0).sort_values(by=['mean_test_score', 'std_test_score'], ascending=False)

    estimators_specs_list, estimators_list = build_estimators_list(
        best_estimators_df=best_estimators_df,
        save_folder=save_folder,
        min_mean_test_score=min_mean_test_score,
        window_size=window_size,
        decimation_factor=decimation_factor,
    )
    model_params_list = [es["estimator"].get_params() for es in estimators_list]

    hp_tot_list_list = []
    sequence_list = []
    
    for _, subject_metadata in metadata.iterrows():
        y_list, hp_tot, invalid_bitmap = predict_samples(
            data_folder,
            estimators_list,
            subject_metadata,
        )
        hp_tot_list_list.append(hp_tot)
        sequence_list.append(build_regressor_sequence(y_list, invalid_bitmap))

        #   hp_tot_list_list =                 y =
        #   [[ 95.0, 90.0, 80.0],              [56,
        #    [ 95.0, 90.0, 80.0],               70,
        #    [ 95.0, 90.0, 80.0],               80,
        #    [ 95.0, 90.0, 80.0]]               34]

    # Alternative
    #for _, subject_metadata in metadata.iterrows():
    #    _, hp_tot, _ = predict_samples(data_folder, estimators_list, subject_metadata)
    #    x.append(hp_tot[0])
    #    y.append(subject_metadata['AHA'])

    X_seq = stack_regressor_sequences(sequence_list)
    y = np.array(metadata['AHA'].values)

    # Organizing data into a dictionary
    data_corrcoef = {
        "Method": estimators_specs_list[0]['method'],
        "Window Size": estimators_specs_list[0]['window_size'],
        "Model Type": estimators_specs_list[0]['model_type'],
        "Gridsearch Hash": estimators_specs_list[0]['gridsearch_hash'],
        "Correlation Coefficient": np.corrcoef(np.array(hp_tot_list_list)[:, 0], y)[0, 1]
    }

    reg_path = 'regressor_' + regressor_hash_from_estimators(
        estimators_list=estimators_list,
        param_grid=REGRESSOR_PARAM_GRID,
    )
    #regressor = BaseEstimator().load_from_path(save_folder + 'Regressors/' + reg_path)
    regressor = jl.load(save_folder + 'Regressors/' + reg_path)
    y_pred = np.asarray(regressor.predict(X_seq), dtype=float)

    data_regression = {
        "Regressor path": reg_path,
        "R2 Score": r2_score(y, y_pred),
        "Classifiers Used": model_params_list
    }

    data_test = {
        "Best Classifier Stats": data_corrcoef,
        "Regressor Stats": data_regression
    }

    # Writing to a JSON file
    with open(save_folder + 'combined_test_stats.json', 'w') as file:
        json.dump(data_test, file, indent=4, default=str)

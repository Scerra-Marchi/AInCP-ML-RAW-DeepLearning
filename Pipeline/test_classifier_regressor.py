import os
import json
import hashlib
import numpy as np
import pandas as pd
import joblib as jl
from sklearn.metrics import r2_score
from sktime.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
from predict_samples import predict_samples
from sklearn.model_selection import cross_val_score

def test_classifier_regressor(data_folder, save_folder, test_indexes, min_mean_test_score, window_size, decimation_factor):

    best_estimators_df = pd.read_csv(save_folder + 'best_estimators_results.csv', index_col=0).sort_values(by=['mean_test_score', 'std_test_score'], ascending=False)

    metadata = pd.read_excel(data_folder + 'metadata2023_08.xlsx')
    metadata.drop(['dom', 'date AHA', 'start AHA', 'stop AHA'], axis=1, inplace=True)   # rimossi 'age_aha', 'gender' 

    estimators_specs_list = [row for index, row in best_estimators_df[(best_estimators_df['mean_test_score'] >= min_mean_test_score) & (best_estimators_df['window_size'] == window_size) & (best_estimators_df['decimation_factor'] == decimation_factor)].iterrows()]

    estimators_list = []
    model_params_list = []
    model_params_concat = ''
    
    for estimators_specs in estimators_specs_list:
        estimator_dir = save_folder + "Trained_models/" + estimators_specs['method'] + "/" + str(estimators_specs['window_size']) + "_points/" + str(estimators_specs['decimation_factor']) + "_decimation_factor/" + estimators_specs['model_type'].split(".")[-1] + "/gridsearch_" + estimators_specs['gridsearch_hash']  + "/"

        with open(estimator_dir + 'GridSearchCV_stats/best_estimator_stats.json', "r") as stats_f:
            grid_search_best_params = json.load(stats_f)

        estimator = BaseEstimator().load_from_path(estimator_dir + 'best_estimator.zip')
        estimators_list.append({'estimator': estimator, 'method': estimators_specs['method'], 'window_size': estimators_specs['window_size'], 'decimation_factor': estimators_specs['decimation_factor'], 'hemi_cluster': grid_search_best_params['Hemi cluster']})
        print('Loaded -> ', estimator_dir + 'best_estimator.zip')
        model_params_concat = model_params_concat + str(estimator.get_params())
        model_params_list.append(estimator.get_params())

    hp_tot_list_list = []
    
    for index in test_indexes:
        _, hp_tot, _, _ = predict_samples(data_folder, estimators_list, metadata['subject'].iloc[index])
        hp_tot_list_list.append(hp_tot)

        #   hp_tot_list_list =                 y =
        #   [[ 95.0, 90.0, 80.0],              [56,
        #    [ 95.0, 90.0, 80.0],               70,
        #    [ 95.0, 90.0, 80.0],               80,
        #    [ 95.0, 90.0, 80.0]]               34]

    # Alternative
    #for i in range (metadata.shape[0]):
    #    _, hp_tot, _, _ = predict_samples(data_folder, estimators_list, metadata['subject'].iloc[i])
    #    x.append(hp_tot[0])
    #    y.append(metadata['AHA'].iloc[i])

    X = np.array(hp_tot_list_list)
    y = np.array(metadata['AHA'].iloc[test_indexes].values)

    # Organizing data into a dictionary
    data_corrcoef = {
        "Method": estimators_specs_list[0]['method'],
        "Window Size": estimators_specs_list[0]['window_size'],
        "Model Type": estimators_specs_list[0]['model_type'],
        "Gridsearch Hash": estimators_specs_list[0]['gridsearch_hash'],
        "Correlation Coefficient": np.corrcoef(np.array(hp_tot_list_list)[:, 0], y)[0, 1]
    }

    reg_path = 'regressor_'+ (hashlib.sha256((model_params_concat).encode()).hexdigest()[:10])
    #regressor = BaseEstimator().load_from_path(save_folder + 'Regressors/' + reg_path)
    regressor = jl.load(save_folder + 'Regressors/' + reg_path)

    data_regression = {
        "Regressor path": reg_path,
        "R2 Score": r2_score(y, regressor.predict(X)),
        "Classifiers Used": model_params_list
    }

    data_test = {
        "Best Classifier Stats": data_corrcoef,
        "Regressor Stats": data_regression
    }

    # Writing to a JSON file
    with open(save_folder + 'combined_test_stats.json', 'w') as file:
        json.dump(data_test, file, indent=4)
import json
import hashlib
import numpy as np
import joblib as jl
import pandas as pd
from sktime.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from predict_samples import predict_samples
import os
import sys

def train_regressor(data_folder, save_folder, train_indexes, min_mean_test_score, window_size, decimation_factor):

    best_estimators_df = pd.read_csv(save_folder + 'best_estimators_results.csv', index_col=0).sort_values(by=['mean_test_score', 'std_test_score'], ascending=False)

    metadata = pd.read_excel(data_folder + 'metadata2023_08.xlsx')
    metadata.drop(['dom', 'date AHA', 'start AHA', 'stop AHA'], axis=1, inplace=True) # rimossi 'age_aha', 'gender'

    # Caricamento dei classificatori

    estimators_specs_list = [row for index, row in best_estimators_df[(best_estimators_df['mean_test_score'] >= min_mean_test_score) & (best_estimators_df['window_size'] == window_size) & (best_estimators_df['decimation_factor'] == decimation_factor)].iterrows()]

    estimators_list = []
    model_params_concat = ''
    
    for estimators_specs in estimators_specs_list:
        estimator_dir = save_folder + "Trained_models/" + estimators_specs['method'] + "/" + str(estimators_specs['window_size']) + "_points/" + str(estimators_specs['decimation_factor']) + "_decimation_factor/" + estimators_specs['model_type'].split(".")[-1] + "/gridsearch_" + estimators_specs['gridsearch_hash']  + "/"

        with open(estimator_dir + 'GridSearchCV_stats/best_estimator_stats.json', "r") as stats_f:
            grid_search_best_params = json.load(stats_f)

        estimator = BaseEstimator().load_from_path(estimator_dir + 'best_estimator.zip')
        estimators_list.append({'estimator': estimator, 'method': estimators_specs['method'], 'window_size': estimators_specs['window_size'], 'decimation_factor': estimators_specs['decimation_factor'], 'hemi_cluster': grid_search_best_params['Hemi cluster']})
        print('Loaded -> ', estimator_dir + 'best_estimator.zip')
        model_params_concat = model_params_concat + str(estimator.get_params())

    # Allenamento del regressore

    hp_tot_list_list = []   # Contiene i CPI calcolati per ogni paziente

    for index in train_indexes:
        print('REGRESSOR: PATIENT ', metadata['subject'].iloc[index], 'BEGIN')
        _,hp_tot_list,_,_ = predict_samples(data_folder, estimators_list, metadata['subject'].iloc[index])
        hp_tot_list_list.append(hp_tot_list)
        print('REGRESSOR: PATIENT ', metadata['subject'].iloc[index], 'END')
        sys.stdout.flush()
        sys.stderr.flush()

    X = np.array(hp_tot_list_list)
    y = np.array(metadata['AHA'].iloc[train_indexes].values)

    reg_path = 'regressor_'+ (hashlib.sha256((model_params_concat).encode()).hexdigest()[:10])

    model = LinearRegression()
    print('REGRESSOR: START FIT')
    model.fit(X, y)
    print('REGRESSOR: END FIT')
    os.makedirs(save_folder + 'Regressors/', exist_ok = True)
    jl.dump(model, save_folder + 'Regressors/'+reg_path)
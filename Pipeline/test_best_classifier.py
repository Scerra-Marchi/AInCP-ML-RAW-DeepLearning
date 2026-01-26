import os
import hashlib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib as jl
from sklearn.metrics import r2_score
from predict_samples import predict_samples

def test_best_classifier(data_folder, save_folder, subjects_indexes):

    best_classifier = pd.read_csv(save_folder + 'best_estimators_results.csv', index_col=0).sort_values(by=['mean_test_score', 'std_test_score'], ascending=False).iloc[0]

    metadata = pd.read_excel(data_folder + 'metadata2023_08.xlsx').iloc[subjects_indexes]
    metadata.drop(['age_aha', 'gender', 'dom', 'date AHA', 'start AHA', 'stop AHA'], axis=1, inplace=True)

    #estimators_specs_list = [row for index, row in best_estimators_df[best_estimators_df['mean_test_score'] >= 0.5].iterrows()]

    estimators_list = []
    model_params_concat = ''

    #for estimators_specs in estimators_specs_list:
    estimator_dir = (
        save_folder
        + "Trained_models/"
        + best_classifier["method"]
        + "/"
        + str(best_classifier["window_size"])
        + "_points/"
        + str(best_classifier.get("decimation_factor", 3))
        + "_decimation_factor/"
        + best_classifier["model_type"].split(".")[-1]
        + "/gridsearch_"
        + best_classifier["gridsearch_hash"]
        + "/"
    )

    estimator = jl.load(estimator_dir + "best_estimator.joblib")
    estimators_list.append(
        {
            "estimator": estimator,
            "method": best_classifier["method"],
            "window_size": best_classifier["window_size"],
            "decimation_factor": best_classifier.get("decimation_factor", 3),
        }
    )
    print("Loaded -> ", estimator_dir + "best_estimator.joblib")
    model_params_concat = model_params_concat + str(estimator.get_params())

    #reg_path = 'Regressors/regressor_'+ (hashlib.sha256((model_params_concat).encode()).hexdigest()[:10])

    x = []
    y = metadata['AHA']

    for index in subjects_indexes:
        _, hp_tot, _, _ = predict_samples(data_folder, estimators_list, index+1)
        x.append(hp_tot[0])


    # Alternative 1
    #for i in range (metadata.shape[0]):
    #    _, hp_tot, _, _ = predict_samples(data_folder, estimators_list, metadata['subject'].iloc[i])
    #    x.append(hp_tot[0])
    #    y.append(metadata['AHA'].iloc[i])

    # Alternative 2
    #for subj in subjects_indexes:
    #    _, hp_tot, _, _ = predict_samples(data_folder, estimators_list, subj)
    #    x.append(hp_tot[0])
    #    y.append(metadata[metadata['subject'] == subj]['AHA'].iloc[0])

        #   hp_tot_list_list =                 y =
        #   [[ 95.0, 90.0, 80.0],              [56,
        #    [ 95.0, 90.0, 80.0],               70,
        #    [ 95.0, 90.0, 80.0],               80,
        #    [ 95.0, 90.0, 80.0]]               34]

    # Corcoef CPI (X) - AHA (y)
    corcoef = np.corrcoef(x, y)
    print(' - corrcoef: ', corcoef)

    # Organizing data into a dictionary
    data = {
        "Method": best_classifier['method'],
        "Window Size": best_classifier['window_size'],
        "Model Type": best_classifier['model_type'],
        "Gridsearch Hash": best_classifier['gridsearch_hash'],
        "Correlation Coefficient": corcoef
    }

    # Writing to a JSON file
    with open(save_folder + 'best_classifier_stats.json', 'w') as file:
        json.dump(data, file, indent=4)

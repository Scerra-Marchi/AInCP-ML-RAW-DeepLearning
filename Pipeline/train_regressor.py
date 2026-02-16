import hashlib
import json
import numpy as np
import joblib as jl
import pandas as pd
from sklearn.linear_model import LinearRegression
from predict_samples import build_estimators_list, predict_samples
import os
import sys


def regressor_hash_from_estimators_specs(estimators_specs_list) -> str:
    specs = sorted(
        [
            (
                str(r["method"]),
                int(r["window_size"]),
                int(r["decimation_factor"]),
                str(r["model_type"]).split(".")[-1],
                str(r["gridsearch_hash"]),
            )
            for r in estimators_specs_list
        ]
    )
    return hashlib.sha256(json.dumps(specs, separators=(",", ":")).encode()).hexdigest()[:10]


def train_regressor(data_folder, save_folder, train_indexes, min_mean_test_score, window_size, decimation_factor):

    best_estimators_df = pd.read_csv(save_folder + 'best_estimators_results.csv', index_col=0).sort_values(by=['mean_test_score', 'std_test_score'], ascending=False)

    metadata = pd.read_excel(data_folder + 'metadata2023_08.xlsx')
    metadata.drop(['dom', 'date AHA', 'start AHA', 'stop AHA'], axis=1, inplace=True) # rimossi 'age_aha', 'gender'

    # Caricamento dei classificatori

    estimators_specs_list, estimators_list = build_estimators_list(
        best_estimators_df=best_estimators_df,
        save_folder=save_folder,
        min_mean_test_score=min_mean_test_score,
        window_size=window_size,
        decimation_factor=decimation_factor,
    )

    reg_path = 'regressor_' + regressor_hash_from_estimators_specs(estimators_specs_list)
    os.makedirs(save_folder + 'Regressors/', exist_ok = True)
    reg_full_path = save_folder + 'Regressors/' + reg_path
    if os.path.exists(reg_full_path):
        print("REGRESSOR: already trained ->", reg_full_path)
        return

    # Allenamento del regressore

    hp_tot_list_list = []   # Contiene i CPI calcolati per ogni paziente

    for index in train_indexes:
        print('REGRESSOR: PATIENT ', metadata['subject'].iloc[index], 'BEGIN')
        _, hp_tot_list, _ = predict_samples(data_folder, estimators_list, index)
        hp_tot_list_list.append(hp_tot_list)
        print('REGRESSOR: PATIENT ', metadata['subject'].iloc[index], 'END')
        sys.stdout.flush()
        sys.stderr.flush()

    X = np.array(hp_tot_list_list)
    y = np.array(metadata['AHA'].iloc[train_indexes].values)

    model = LinearRegression()
    print('REGRESSOR: START FIT')
    model.fit(X, y)
    print('REGRESSOR: END FIT')
    jl.dump(model, reg_full_path)

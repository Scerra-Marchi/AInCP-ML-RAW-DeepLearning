import json
import numpy as np
import pandas as pd
import joblib as jl
from sklearn.metrics import r2_score
from predict_samples import build_estimators_list, predict_samples
from train_regressor import (
    build_regressor_sample,
    ffnn_model_path,
    regressor_model_path,
)

def test_classifier_regressor(
    data_folder,
    save_folder,
    metadata,
    min_mean_test_score=None,
    window_size=None,
    decimation_factor=1,
):
    best_estimators_df = pd.read_csv(save_folder + 'best_estimators_results.csv', index_col=0).sort_values(by=['mean_test_score', 'std_test_score'], ascending=False)

    estimators_specs_list, estimators_list = build_estimators_list(
        best_estimators_df=best_estimators_df,
        save_folder=save_folder,
        min_mean_test_score=min_mean_test_score,
        window_size=window_size,
        decimation_factor=decimation_factor,
    )
    if not estimators_list:
        raise ValueError("No classifiers selected for testing with the current filters.")

    model_params_list = [es["estimator"].get_params() for es in estimators_list]

    hp_tot_list_list = []
    hard_pred_tot_list_list = []
    raw_inputs = []
    
    for _, subject_metadata in metadata.iterrows():
        y_list, hp_tot, invalid_bitmap = predict_samples(
            data_folder,
            estimators_list,
            subject_metadata,
        )
        hp_tot_list_list.append(hp_tot)
        raw_inputs.append(
            build_regressor_sample(
                data_folder,
                estimators_list,
                subject_metadata,
                input_type="week",
                predictions_list=y_list,
                invalid_bitmap=invalid_bitmap,
            )
        )
        invalid_bitmap = np.asarray(invalid_bitmap, dtype=np.uint8)
        invalid_mask = np.asarray(invalid_bitmap, dtype=bool)
        hard_pred_tot_list_list.append([
            float((np.asarray(probs, dtype=float)[~invalid_mask] >= 0.5).mean())
            for probs in y_list
        ])

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

    X_raw = np.asarray(raw_inputs, dtype=object)
    y = np.array(metadata['AHA'].values, dtype=float)

    hp_tot_array = np.asarray(hp_tot_list_list, dtype=float)
    hard_pred_tot_array = np.asarray(hard_pred_tot_list_list, dtype=float)

    classifiers_stats = []
    for i, estimators_specs in enumerate(estimators_specs_list):
        classifiers_stats.append(
            {
                "Classifier Index": i,
                "Method": estimators_specs["method"],
                "Window Size": int(estimators_specs["window_size"]),
                "Decimation Factor": int(estimators_specs["decimation_factor"]),
                "Model Type": estimators_specs["model_type"],
                "Gridsearch Hash": str(estimators_specs["gridsearch_hash"]),
                "Hard Prediction Threshold": 0.5,
                "Correlation Mean Probability vs AHA": float(np.corrcoef(hp_tot_array[:, i], y)[0, 1]),
                "Correlation Mean Hard Prediction vs AHA": float(np.corrcoef(hard_pred_tot_array[:, i], y)[0, 1]),
            }
        )

    model_path = regressor_model_path(
        save_folder=save_folder,
        estimators_list=estimators_list,
    )
    gru_regressor = jl.load(model_path)
    y_pred_gru = np.asarray(gru_regressor.predict(X_raw), dtype=float).reshape(-1)

    ffnn_path = ffnn_model_path(
        save_folder=save_folder,
        estimators_list=estimators_list,
    )
    ffnn_regressor = jl.load(ffnn_path)
    y_pred_ffnn = np.asarray(ffnn_regressor.predict(X_raw), dtype=float).reshape(-1)

    data_regression = {
        "Regressor path": model_path,
        "R2 Score": r2_score(y, y_pred_gru),
        "Classifiers Used": model_params_list
    }
    data_ffnn = {
        "FFNN path": ffnn_path,
        "R2 Score": r2_score(y, y_pred_ffnn),
    }

    data_test = {
        "Selected Classifiers Stats": classifiers_stats,
        "GRU Regressor Stats": data_regression,
        "FFNN Regressor Stats": data_ffnn,
    }

    # Writing to a JSON file
    with open(save_folder + 'combined_test_stats.json', 'w') as file:
        json.dump(data_test, file, indent=4, default=str)

import numpy as np
from create_windows import create_windows


def predict_samples(data_folder, estimators, subject_indexes):

    if not estimators:
        raise ValueError("You have selected zero estimators to predict the samples with")

    if len(set(es['window_size'] for es in estimators)) != 1:
        raise ValueError("You have selected estimators that operate on different window sizes")

    window_size = estimators[0]['window_size']
    decimation_factor = estimators[0]['decimation_factor']

    # ===============================
    # FEATURE CACHE PER METHOD
    # ===============================
    method_to_features = {}
    invalid_bitmap = None

    unique_methods = set(es['method'] for es in estimators)

    for method in unique_methods:

        X, _, _, _, invalid_bitmap = create_windows(
            data_folder=data_folder,
            subjects_indexes=subject_indexes,
            operation_type=method,
            WINDOW_SIZE=window_size,
            decimation_factor=decimation_factor,
            input_type='week',
        )

        method_to_features[method] = X

    # assegna le serie a ogni estimator
    for es in estimators:
        es['series'] = method_to_features[es['method']]

    # ===============================
    # PREDIZIONE
    # ===============================
    y_list = []        # probabilità per finestra (0 sulle finestre invalide)
    hp_tot_list = []   # media delle probabilità per paziente

    invalid_mask = np.asarray(invalid_bitmap, dtype=bool)

    for es in estimators:

        X = es['series']

        # ===============================
        # PROBABILITÀ POST-SIGMOIDE
        # ===============================
        # predict_proba -> (N, 2)
        # [:, 1] = P(classe positiva = sano)
        probs = es['estimator'].predict_proba(X)[:, 1]

        # finestre valide
        valid_probs = probs[~invalid_mask]

        # ===============================
        # PROBABILITÀ PER FINESTRA (con bitmap)
        # ===============================
        probs_with_bitmap = probs.copy()
        probs_with_bitmap[invalid_mask] = 0.0
        y_list.append(probs_with_bitmap)

        # ===============================
        # MEDIA PROBABILITÀ PER TIME SERIES
        # ===============================
        if valid_probs.size > 0:
            hp_tot = float(np.mean(valid_probs))
        else:
            hp_tot = np.nan

        hp_tot_list.append(hp_tot)

    return y_list, hp_tot_list

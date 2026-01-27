import numpy as np
from create_windows import create_windows

def predict_samples(data_folder, estimators, patient):

    if not estimators:
        raise ValueError("You have selected zero estimators to predict the samples with")

    if len(set(es['window_size'] for es in estimators)) != 1:
        raise ValueError("You have selected estimators that operate on different window sizes")

    window_size = estimators[0]['window_size']
    decimation_factor = estimators[0]['decimation_factor']

    subject_indexes = [patient]

    # ===============================
    # FEATURE CACHE PER METHOD
    # ===============================
    method_to_features = {}
    mag_D = mag_ND = invalid_bitmap = None

    unique_methods = set(es['method'] for es in estimators)

    for method in unique_methods:

        X, _, _, _, mag_D, mag_ND, invalid_bitmap = create_windows(
            data_folder=data_folder,
            subjects_indexes=subject_indexes,
            operation_type=method,
            WINDOW_SIZE=window_size,
            decimation_factor=decimation_factor,
            input_type='week',
            return_mag=1
        )

        method_to_features[method] = X

    # assegna le serie a ogni estimator
    for es in estimators:
        es['series'] = method_to_features[es['method']]

    # ===============================
    # PREDIZIONE
    # ===============================
    y_list = []
    hp_tot_list = []

    for es in estimators:

        X = es['series']
        y = np.asarray(es['estimator'].predict(X))

        cluster_healthy_samples = int(np.sum(y == 0))     # Non emiplegici
        cluster_hemiplegic_samples = int(np.sum(y == 1))  # Emiplegici
        # Apply bitmap: overwrite invalid windows with 0
        y[np.asarray(invalid_bitmap, dtype=bool)] = -1
        y_mapped = np.zeros_like(y, dtype=int)
        y_mapped[y == 0] = 1
        y_mapped[y == 1] = -1
        y[np.asarray(invalid_bitmap, dtype=bool)] = 0
        y_list.append(y_mapped)

        if (cluster_healthy_samples + cluster_hemiplegic_samples) > 0:
            hp_tot = (
                cluster_healthy_samples /
                (cluster_healthy_samples + cluster_hemiplegic_samples)
            ) * 100
        else:
            hp_tot = np.nan

        hp_tot_list.append(hp_tot)

    return y_list, hp_tot_list, mag_D, mag_ND

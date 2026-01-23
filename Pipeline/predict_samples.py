import numpy as np
from create_windows import create_windows

def predict_samples(data_folder, estimators, patient):
    
    import time
    start = time.perf_counter()

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
    mag_D = mag_ND = None

    unique_methods = set(es['method'] for es in estimators)

    for method in unique_methods:

        X, _, _, _, mD, mND = create_windows(
            data_folder=data_folder,
            subjects_indexes=subject_indexes,
            operation_type=method,
            WINDOW_SIZE=window_size,
            decimation_factor=decimation_factor,
            input_type='week',
            return_mag=1
        )

        method_to_features[method] = X
        mag_D = mD
        mag_ND = mND

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
        y = es['estimator'].predict(X)

        hemi_cluster = es['hemi_cluster']

        cluster_healthy_samples = 0
        cluster_hemiplegic_samples = 0

        for k in range(len(y)):
            if y[k] == hemi_cluster:
                cluster_hemiplegic_samples += 1
                y[k] = -1
            else:
                cluster_healthy_samples += 1
                y[k] = 1

        y_list.append(y)

        if (cluster_healthy_samples + cluster_hemiplegic_samples) > 0:
            hp_tot = (
                cluster_healthy_samples /
                (cluster_healthy_samples + cluster_hemiplegic_samples)
            ) * 100
        else:
            hp_tot = np.nan

        hp_tot_list.append(hp_tot)
    
    end = time.perf_counter()
    print(f"ELAPSED TIME FOR PREDICT_SAMPLES                                                        : {end - start:.4f} s")

    return y_list, hp_tot_list, mag_D, mag_ND

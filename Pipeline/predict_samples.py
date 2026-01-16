import numpy as np
import pandas as pd
from elaborate_magnitude import elaborate_magnitude
from create_windows import decimate_df

def predict_samples(data_folder, estimators, patient):

    if not estimators:
        print('You have selected zero estimators to predict the samples with')
        exit(1)

    if len(set(es['window_size'] for es in estimators)) != 1:
        print('You have selected estimators that operates on different window sizes')
        exit(1)

    df = pd.read_csv(data_folder + 'week/' + str(patient) + '_week_RAW.csv')
    df = decimate_df(df, estimators[0]['decimation_factor']) # Decimazione delle time series 
    magnitude_D = np.sqrt(np.square(np.array(df['x_D'])) + np.square(np.array(df['y_D'])) + np.square(np.array(df['z_D'])))
    magnitude_ND = np.sqrt(np.square(np.array(df['x_ND'])) + np.square(np.array(df['y_ND'])) + np.square(np.array(df['z_ND'])))

    to_discard = []

    window_size = estimators[0]['window_size']

    for es in estimators:
        es['series'] = []   # Nuovo campo che conterrà le finestre

    # Fase di chunking
    for j in range (0, len(magnitude_D), window_size):

        chunk_D = magnitude_D[j:j + window_size]
        chunk_ND = magnitude_ND[j:j + window_size]

        if chunk_D.size == window_size and chunk_ND.size == window_size:
            
            for es in estimators:
                es['series'].append(elaborate_magnitude(es['method'], chunk_D, chunk_ND))

            if np.all(chunk_D == 0) and np.all(chunk_ND == 0):  # Vengono scartate le finestre con solo valori a zero
                to_discard.append(int(j/window_size))

    y_list = [] # Lista dei valori predetti da ogni classificatore
    hp_tot_list = [] # Lista dei CPI calcolati per questo paziente

    # Fase di predizione
    for es in estimators:
        #print(np.array(es['series']).shape)
        y = es['estimator'].predict(np.array(es['series']))
        #print(es['method'])

        for index in to_discard:
            y[index] = -1

        hemi_cluster = es['hemi_cluster']

        cluster_healthy_samples = 0     # Non emiplegici
        cluster_hemiplegic_samples = 0  # Emiplegici

        for k in range(len(y)):
            if y[k] == hemi_cluster:
                cluster_hemiplegic_samples += 1
                y[k] = -1
            elif y[k] != -1:
                cluster_healthy_samples += 1  
                y[k] = 1
            else:
                y[k] = 0
        
        y_list.append(y)

        # Calcolo del CPI
        hp_tot = np.nan
        if (cluster_healthy_samples != 0 or cluster_hemiplegic_samples != 0):
            hp_tot = (cluster_healthy_samples / (cluster_hemiplegic_samples + cluster_healthy_samples)) * 100

        hp_tot_list.append(hp_tot)

    return y_list, hp_tot_list, magnitude_D, magnitude_ND
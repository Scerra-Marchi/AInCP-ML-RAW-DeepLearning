import pandas as pd
import math
import numpy as np
from elaborate_magnitude import elaborate_magnitude
from scipy.signal import decimate

def decimate_df(data, factor):
    if factor <= 1:
        return data
    # Isolare le colonne numeriche
    df_axis = data[['x_D', 'y_D', 'z_D', 'x_ND', 'y_ND', 'z_ND']]
    # Eseguire il downsampling
    df_decimated = pd.DataFrame(decimate(df_axis, factor, axis=0, ftype='fir', zero_phase=True), columns=df_axis.columns).reset_index(drop=True)
    # Isolare i timestamps corrispondenti
    timestamps = pd.DataFrame(data['datetime'].iloc[::factor].reset_index(drop=True))
    assert timestamps.shape[0] == df_decimated.shape[0], "Mismatch in lunghezza dopo decimation."
    # Creare un nuovo DataFrame con i dati decimati e i timestamps
    return pd.concat([timestamps, df_decimated], axis=1)

def create_windows(data_folder, subjects_indexes, operation_type, WINDOW_SIZE, decimation_factor):
    series = []
    y_AHA = []
    y_MACS =[]
    y = []
    metadata = pd.read_excel(data_folder + 'metadata2023_08.xlsx').iloc[subjects_indexes]

    for index in range (metadata.shape[0]):
        df = pd.read_csv(data_folder + 'AHA/' + str(metadata['subject'].iloc[index]) + '_AHA_RAW.csv')

        # Si fa il downsampling della time series, prendendo un campione ogni 3 (80 Hz -> 26.67 Hz)
        df = decimate_df(df, decimation_factor)

        # Nel caso in cui non bastasse una duplicazione dell'intera time series questa verrà scartata
        if df.shape[0]<WINDOW_SIZE:
            df_concat = pd.concat([df, df.iloc[:WINDOW_SIZE-df.shape[0]]], ignore_index = True, axis = 0)
            df = df_concat
            #print('MODIFICATO Paziente ' + str(j) + ' -> df.shape[0] = ' + str(df.shape[0]))

        scart = (df.shape[0] % WINDOW_SIZE)/2
        
        df_cut = df.iloc[math.ceil(scart):df.shape[0]-math.floor(scart)]

        x_D = np.array(df_cut['x_D'])
        y_D = np.array(df_cut['y_D'])
        z_D = np.array(df_cut['z_D'])

        x_ND = np.array(df_cut['x_ND'])
        y_ND = np.array(df_cut['y_ND'])
        z_ND = np.array(df_cut['z_ND'])

        # Calculating magnitude
        magnitude_D = np.sqrt(np.square(x_D) + np.square(y_D) + np.square(z_D))
        magnitude_ND = np.sqrt(np.square(x_ND) + np.square(y_ND) + np.square(z_ND))
        for i in range (0, len(magnitude_D), WINDOW_SIZE):
            chunk_D = magnitude_D[i:i + WINDOW_SIZE]
            chunk_ND = magnitude_ND[i:i + WINDOW_SIZE]
            series.append(elaborate_magnitude(operation_type, chunk_D, chunk_ND))
            y_AHA.append(metadata['AHA'].iloc[index])
            y_MACS.append(metadata['MACS'].iloc[index])
            y.append(metadata['hemi'].iloc[index]-1)
    
    return np.array(series), y_AHA, y_MACS, np.array(y)

    #return np.array(copy.deepcopy(series)), y_AHA, y_MACS, np.array(copy.deepcopy(y))
    # create a list of dictionaries
    #dicts = [{"column": lst} for lst in series]
    #return pd.DataFrame(dicts).copy(), y_AHA, y_MACS, np.array(y)
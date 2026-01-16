import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import joblib as jl
from create_windows import decimate_df
import numpy as np
import math

def calculate_ENMO(data_folder, patient, ds_freq):
    df = pd.read_csv(data_folder + 'week/' + str(patient) + '_week_RAW.csv')
    df = decimate_df(df, 3) # Decimazione delle time series a 26.67Hz
    magnitude_D = np.sqrt(np.square(np.array(df['x_D'])) + np.square(np.array(df['y_D'])) + np.square(np.array(df['z_D'])))
    magnitude_ND = np.sqrt(np.square(np.array(df['x_ND'])) + np.square(np.array(df['y_ND'])) + np.square(np.array(df['z_ND'])))
    # Si calcola l'ENMO per ogni valore
    enmo_D = np.maximum(magnitude_D - 1, 0)
    enmo_ND = np.maximum(magnitude_ND - 1, 0)
    # Si controlla che l'ENMO non abbia valori negativi
    if np.any(enmo_D < 0) or np.any(enmo_ND < 0):
        raise ValueError("ENMO ha valori negativi")
    # Si media su finestre di 1 secondo
    one_sec_window =  math.ceil(ds_freq) # Numero di campioni in un secondo
    enmo_D_windows = np.array([enmo_D[n:n+one_sec_window].mean() for n in range(0, len(enmo_D), one_sec_window)])
    enmo_ND_windows = np.array([enmo_ND[n:n+one_sec_window].mean() for n in range(0, len(enmo_ND), one_sec_window)])
    return enmo_D_windows, enmo_ND_windows


def plot_AI_raw(data_folder, save_folder, indexes):

    metadata = pd.read_excel(data_folder + 'metadata2023_08.xlsx').iloc[indexes]
    metadata.drop(['dom', 'date AHA', 'start AHA', 'stop AHA'], axis=1, inplace=True)

    timestamps = jl.load('timestamps_list_1sec') 

    block_samples = int(6 * 60 * 60)

    for index in range(len(metadata)):
        subject = metadata['subject'].iloc[index]
        enmo_D, enmo_ND = calculate_ENMO(data_folder, subject, 26.67)
        ai_list = []
        subList_enmoD = [enmo_D[n:n+block_samples] for n in range(0, len(enmo_D), block_samples)]
        subList_enmoND = [enmo_ND[n:n+block_samples] for n in range(0, len(enmo_ND), block_samples)]
        for l in range(len(subList_enmoD)):
            if (subList_enmoD[l].mean() + subList_enmoND[l].mean()) == 0:
                ai_list.append(np.nan)
            else:
                ai_list.append(((subList_enmoD[l].mean() - subList_enmoND[l].mean()) / (subList_enmoD[l].mean() + subList_enmoND[l].mean())) * 100)

        print("ai_list length:", len(ai_list))
        print("timestamps[::block_samples] length:", len(timestamps[::block_samples]))
        print("ai_list:", ai_list)

        plt.xlabel("Orario")
        plt.ylabel("Asimmetry Index")
        plt.grid()
        ax = plt.gca()
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        # Si imposta il limite y tra -100 e 100
        plt.ylim(-100, 100)
        plt.plot(timestamps[::block_samples], ai_list)
        plt.gcf().set_size_inches(8, 2)
        plt.tight_layout()
        plt.savefig(save_folder + '/subject_' +str(subject)+'_AI_raw.png', dpi = 500)
        plt.close()


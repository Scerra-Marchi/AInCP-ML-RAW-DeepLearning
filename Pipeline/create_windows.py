import pandas as pd
import math
import numpy as np
from scipy.signal import decimate
from elaborate_magnitude import elaborate_magnitude


def decimate_df(data, factor):
    if factor <= 1:
        return data

    df_axis = data[['x_D', 'y_D', 'z_D', 'x_ND', 'y_ND', 'z_ND']]

    df_decimated = pd.DataFrame(
        decimate(df_axis, factor, axis=0, ftype='fir', zero_phase=True),
        columns=df_axis.columns
    ).reset_index(drop=True)

    timestamps = data[['datetime']].iloc[::factor].reset_index(drop=True)

    assert len(df_decimated) == len(timestamps), "Mismatch after decimation"

    return pd.concat([timestamps, df_decimated], axis=1)


def create_windows(data_folder, subjects_indexes, operation_type, WINDOW_SIZE, decimation_factor):

    series = []
    y_AHA = []
    y_MACS = []
    y = []

    metadata = pd.read_excel(
        data_folder + 'metadata2023_08.xlsx'
    ).iloc[subjects_indexes].reset_index(drop=True)

    for index in range(metadata.shape[0]):

        subject = metadata.loc[index, 'subject']
        df = pd.read_csv(f"{data_folder}AHA/{subject}_AHA_RAW.csv")

        df = decimate_df(df, decimation_factor)

        if len(df) < WINDOW_SIZE:
            df = pd.concat(
                [df, df.iloc[:WINDOW_SIZE - len(df)]],
                ignore_index=True
            )

        scart = (len(df) % WINDOW_SIZE) // 2
        df_cut = df.iloc[scart:len(df) - scart]

        # === Magnitude computation (vectorized) ===
        D = df_cut[['x_D', 'y_D', 'z_D']].to_numpy()
        ND = df_cut[['x_ND', 'y_ND', 'z_ND']].to_numpy()

        magnitude_D = np.linalg.norm(D, axis=1)
        magnitude_ND = np.linalg.norm(ND, axis=1)

        # === Chunking vettoriale ===
        n_windows = len(magnitude_D) // WINDOW_SIZE

        magnitude_D = magnitude_D.reshape(n_windows, WINDOW_SIZE)
        magnitude_ND = magnitude_ND.reshape(n_windows, WINDOW_SIZE)

        # === Elaborazione batch ===
        features = elaborate_magnitude(
            operation_type,
            magnitude_D,
            magnitude_ND
        )

        series.append(features)

        y_AHA.extend([metadata.loc[index, 'AHA']] * n_windows)
        y_MACS.extend([metadata.loc[index, 'MACS']] * n_windows)
        y.extend([metadata.loc[index, 'hemi'] - 1] * n_windows)

    return (
        np.vstack(series),
        np.array(y_AHA),
        np.array(y_MACS),
        np.array(y)
    )

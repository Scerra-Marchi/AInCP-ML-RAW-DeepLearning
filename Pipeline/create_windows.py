import pandas as pd
import math
import numpy as np
from scipy.signal import decimate
from elaborate_magnitude import elaborate_magnitude


def decimate_df(data, factor):
    if factor <= 1:
        return data

    df_decimated = pd.DataFrame(
        decimate(data, factor, axis=0, ftype='fir', zero_phase=True),
        columns=data.columns
    ).reset_index(drop=True)

    return df_decimated


def create_windows(
    data_folder,
    subjects_indexes,
    operation_type,
    WINDOW_SIZE,
    decimation_factor,
    input_type='AHA',
    return_mag=0
):

    series = []
    y_AHA = []
    y_MACS = []
    y = []
    mag_D = np.array([])
    mag_ND = np.array([])
    invalid_bitmap = []


    metadata = pd.read_excel(
        data_folder + 'metadata2023_08.xlsx'
    ).iloc[subjects_indexes].reset_index(drop=True)

    for index in range(metadata.shape[0]):

        subject = metadata.loc[index, 'subject']
        df = pd.read_csv(f"{data_folder}{input_type}/{subject}_{input_type}_RAW.csv", usecols=["x_D", "y_D", "z_D",'x_ND', 'y_ND', 'z_ND'], engine="pyarrow")

        df = decimate_df(df, decimation_factor)

        if len(df) < WINDOW_SIZE:
            df = pd.concat(
                [df, df.iloc[:WINDOW_SIZE - len(df)]],
                ignore_index=True
            )

        usable_len = (len(df) // WINDOW_SIZE) * WINDOW_SIZE
        df_cut = df.iloc[:usable_len]


        # === Magnitude computation (vectorized) ===
        D = df_cut[['x_D', 'y_D', 'z_D']].to_numpy()
        ND = df_cut[['x_ND', 'y_ND', 'z_ND']].to_numpy()

        # A window is invalid if ALL 6 channels (D and ND) are exactly 0 for the whole WINDOW_SIZE
        n_windows = len(D) // WINDOW_SIZE
        zero_samples = np.all(D == 0, axis=1) & np.all(ND == 0, axis=1)
        invalid_windows = zero_samples.reshape(n_windows, WINDOW_SIZE).all(axis=1)
        invalid_bitmap.extend(invalid_windows.tolist())

        magnitude_D = np.linalg.norm(D, axis=1)
        magnitude_ND = np.linalg.norm(ND, axis=1)
        
        if(return_mag):
            mag_D = np.copy(magnitude_D)
            mag_ND = np.copy(magnitude_ND)

        # === Chunking vettoriale ===
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
        np.array(y),
        mag_D,
        mag_ND,
        np.asarray(invalid_bitmap, dtype=np.uint8)
    )

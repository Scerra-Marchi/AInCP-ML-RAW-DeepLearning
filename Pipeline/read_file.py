import pandas as pd
import numpy as np
from scipy.signal import decimate


def decimate_df(data, factor):
    if factor <= 1:
        return data

    df_decimated = pd.DataFrame(
        decimate(data, factor, axis=0, ftype='fir', zero_phase=True),
        columns=data.columns
    ).reset_index(drop=True)

    return df_decimated


def read_file(
    data_folder,
    subject,
    WINDOW_SIZE,
    decimation_factor,
    input_type='AHA',
    return_enmo=0
):
    
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
    
    if(return_enmo):
        D = np.maximum(np.linalg.norm(D, axis=1) - 1.0, 0.0)
        ND =  np.maximum(np.linalg.norm(ND, axis=1) - 1.0, 0.0)

    return D, ND

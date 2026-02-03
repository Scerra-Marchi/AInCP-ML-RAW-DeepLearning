import pandas as pd
import numpy as np
from elaborate_magnitude import elaborate_magnitude
from read_file import read_file


def create_windows(
    data_folder,
    subjects_indexes,
    operation_type,
    WINDOW_SIZE,
    decimation_factor,
    input_type='AHA',
):

    series = []
    y_AHA = []
    y_MACS = []
    y = []
    invalid_bitmap = []

    metadata = pd.read_excel(
        data_folder + 'metadata2023_08.xlsx'
    ).iloc[subjects_indexes].reset_index(drop=True)

    for index in range(metadata.shape[0]):

        D, ND = read_file(data_folder,
                          metadata.loc[index, 'subject'],
                          WINDOW_SIZE,
                          decimation_factor,
                          input_type=input_type)
        
        # A window is invalid if ALL 6 channels (D and ND) are exactly 0 for the whole WINDOW_SIZE
        n_windows = len(D) // WINDOW_SIZE
        zero_samples = np.all(D == 0, axis=1) & np.all(ND == 0, axis=1)
        invalid_windows = zero_samples.reshape(n_windows, WINDOW_SIZE).all(axis=1)
        invalid_bitmap.extend(invalid_windows.tolist())

        D_w = D.reshape(n_windows, WINDOW_SIZE, 3)
        ND_w = ND.reshape(n_windows, WINDOW_SIZE, 3)

        features = elaborate_magnitude(
            operation_type,
            D_w,
            ND_w
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
        np.asarray(invalid_bitmap, dtype=np.uint8)
    )

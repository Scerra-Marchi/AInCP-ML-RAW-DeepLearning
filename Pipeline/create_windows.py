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
    std_tol=0.005  # soglia std per considerare una finestra "ferma" per ciascuna feature
):

    series = []
    y_AHA = []
    y_MACS = []
    y = []

    per_subject_data = []

    metadata = pd.read_excel(
        data_folder + 'metadata2023_08.xlsx'
    ).iloc[subjects_indexes].reset_index(drop=True)

    # ==========================================================
    # PASS 1 → creazione finestre raw per ciascun soggetto
    # ==========================================================

    for index in range(metadata.shape[0]):

        # Legge i dati raw dei due polsi
        D, ND = read_file(
            data_folder,
            metadata.loc[index, 'subject'],
            WINDOW_SIZE,
            decimation_factor,
            input_type=input_type
        )

        n_windows = len(D) // WINDOW_SIZE

        D_w = D.reshape(n_windows, WINDOW_SIZE, 3)
        ND_w = ND.reshape(n_windows, WINDOW_SIZE, 3)

        per_subject_data.append((D_w, ND_w))

    # ==========================================================
    # PASS 2 → costruzione bitmap finestre non significative
    # ==========================================================

    invalid_bitmap = []

    for index, (D_w, ND_w) in enumerate(per_subject_data):

        n_windows = D_w.shape[0]

        # Combiniamo le due matrici in shape [windows, WINDOW_SIZE, 6]
        combined = np.concatenate([D_w, ND_w], axis=2)

        # Calcola std per ciascuna finestra e ciascuna feature
        std_features = np.std(combined, axis=1)  # shape [windows, 6]

        # Una finestra è non significativa se **tutte le feature** hanno std < soglia
        invalid_windows = np.all(std_features < std_tol, axis=1)
        invalid_bitmap.extend(invalid_windows.tolist())

        # Costruisci features
        features = elaborate_magnitude(
            operation_type,
            D_w,
            ND_w
        )

        series.append(features)

        y_AHA.extend([metadata.loc[index, 'AHA']] * n_windows)
        y_MACS.extend([metadata.loc[index, 'MACS']] * n_windows)
        # Binary target for classifier:
        # healthy -> 1 when AHA is exactly 100, otherwise hemiplegic/not-healthy -> 0.
        target = int(metadata.loc[index, 'AHA'] == 100)
        y.extend([target] * n_windows)

    return (
        np.vstack(series),
        np.array(y_AHA),
        np.array(y_MACS),
        np.array(y),
        np.asarray(invalid_bitmap, dtype=np.uint8)
    )

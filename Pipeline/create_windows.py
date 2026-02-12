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

    all_energies = []
    per_subject_data = []

    metadata = pd.read_excel(
        data_folder + 'metadata2023_08.xlsx'
    ).iloc[subjects_indexes].reset_index(drop=True)

    # ==========================================================
    # PASS 1 → calcolo energia per tutte le finestre
    # ==========================================================

    for index in range(metadata.shape[0]):

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

        # Magnitudo calcolata dai raw
        mag_D = np.sqrt(np.sum(D_w ** 2, axis=2))
        mag_ND = np.sqrt(np.sum(ND_w ** 2, axis=2))

        # Energia finestra = somma delle deviazioni standard
        energy = np.std(mag_D, axis=1) + np.std(mag_ND, axis=1)

        all_energies.append(energy)
        per_subject_data.append((D_w, ND_w, energy))

    # Flatten
    all_energies = np.concatenate(all_energies)

    # Scala globale robusta
    global_median_energy = np.median(all_energies)

    # Protezione da dataset totalmente piatto
    if global_median_energy == 0:
        global_median_energy = 1e-12

    threshold = 1e-6 * global_median_energy

    # ==========================================================
    # PASS 2 → costruzione dataset + bitmap
    # ==========================================================

    invalid_bitmap = []

    for index, (D_w, ND_w, energy) in enumerate(per_subject_data):

        n_windows = D_w.shape[0]

        invalid_windows = energy < threshold
        invalid_bitmap.extend(invalid_windows.tolist())

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

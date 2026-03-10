import numpy as np
from read_file import read_file
from signal_features import build_signal_features, compute_invalid_bitmap


def create_windows(
    data_folder,
    metadata,
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
    subjects = []

    per_subject_data = []

    # ==========================================================
    # PASS 1 → creazione finestre raw per ciascun soggetto
    # ==========================================================

    for _, subject_metadata in metadata.iterrows():
        subject = int(subject_metadata['subject'])

        # Legge i dati raw dei due polsi
        D, ND = read_file(
            data_folder,
            subject,
            WINDOW_SIZE,
            decimation_factor,
            input_type=input_type
        )

        n_windows = len(D) // WINDOW_SIZE

        D_w = D.reshape(n_windows, WINDOW_SIZE, 3)
        ND_w = ND.reshape(n_windows, WINDOW_SIZE, 3)

        per_subject_data.append((subject_metadata, D_w, ND_w))

    # ==========================================================
    # PASS 2 → costruzione bitmap finestre non significative
    # ==========================================================

    invalid_bitmap = []

    for subject_metadata, D_w, ND_w in per_subject_data:

        n_windows = D_w.shape[0]

        # Combiniamo le due matrici in shape [windows, WINDOW_SIZE, 6]
        # Una finestra è non significativa se **tutte le feature** hanno std < soglia
        invalid_windows = compute_invalid_bitmap(D_w, ND_w, std_tol=std_tol)
        invalid_bitmap.extend(invalid_windows.tolist())

        # Costruisci features
        features = build_signal_features(
            operation_type,
            D_w,
            ND_w
        )

        series.append(features)

        y_AHA.extend([subject_metadata['AHA']] * n_windows)
        y_MACS.extend([subject_metadata['MACS']] * n_windows)
        # Binary target for classifier:
        # healthy -> 1 when AHA is exactly 100, otherwise hemiplegic/not-healthy -> 0.
        target = int(subject_metadata['AHA'] == 100)
        y.extend([target] * n_windows)
        subjects.extend([int(subject_metadata['subject'])] * n_windows)

    return (
        np.vstack(series),
        np.array(y_AHA),
        np.array(y_MACS),
        np.array(y),
        np.asarray(invalid_bitmap, dtype=np.uint8),
        np.array(subjects),
    )

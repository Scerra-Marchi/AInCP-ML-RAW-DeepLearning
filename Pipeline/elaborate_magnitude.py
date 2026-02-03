import numpy as np

def elaborate_magnitude(operation_type, D, ND):
    """
    D, ND: shape (n_windows, WINDOW_SIZE, 3)
    """

    # Magnitude (lazy: solo se serve)
    if operation_type in ('concat', 'difference', 'ai', 'enmo'):
        mag_D = np.linalg.norm(D, axis=2)
        mag_ND = np.linalg.norm(ND, axis=2)

    if operation_type == 'concat':
        return np.concatenate((mag_D, mag_ND), axis=1)

    elif operation_type == 'difference':
        return mag_D - mag_ND

    elif operation_type == 'ai':
        denom = mag_D + mag_ND
        out = np.zeros_like(mag_D)
        mask = denom != 0
        np.divide(
            mag_D - mag_ND,
            denom,
            where=mask,
            out=out
        )
        return out * 100

    # ===== ENMO =====
    elif operation_type == 'enmo':
        enmo_D = np.maximum(mag_D - 1.0, 0.0)
        enmo_ND = np.maximum(mag_ND - 1.0, 0.0)
        return np.stack((enmo_D, enmo_ND), axis=2)
        # shape: (n_windows, WINDOW_SIZE, 2)

    # ===== RAW =====
    elif operation_type == 'raw':
        return np.concatenate((D, ND), axis=2)
        # shape: (n_windows, WINDOW_SIZE, 6)

    else:
        raise ValueError(f"operation type non supportata: {operation_type}")

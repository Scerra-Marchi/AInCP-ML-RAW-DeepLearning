import numpy as np

def elaborate_magnitude(operation_type, magnitude_D, magnitude_ND):
    """
    magnitude_D, magnitude_ND: shape (n_windows, WINDOW_SIZE)
    """

    if operation_type == 'concat':
        # output shape: (n_windows, 2 * WINDOW_SIZE)
        return np.concatenate((magnitude_D, magnitude_ND), axis=1)

    elif operation_type == 'difference':
        # shape: (n_windows, WINDOW_SIZE)
        return magnitude_D - magnitude_ND

    elif operation_type == 'ai':
        denom = magnitude_D + magnitude_ND
        out = np.zeros_like(magnitude_D)
        mask = denom != 0
        np.divide(
            magnitude_D - magnitude_ND,
            denom,
            where=mask,
            out=out
        )
        return out * 100

    else:
        raise ValueError("operation type non supportata")

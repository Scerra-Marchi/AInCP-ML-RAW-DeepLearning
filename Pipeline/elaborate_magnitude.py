import numpy as np

_EPSILON = 1e-6
_MAGNITUDE_TYPES = {
    "concat",
    "difference",
    "ai",
    "magnitude",
    "enmo",
    "raw_enmo_ai",
    "raw_enmo",
    "raw_ai",
    "raw_ratio",
    "raw_jerk",
    "raw_enmo_ai_ratio_jerk",
    "enmo_asymmetry",
}


def _asymmetry_index(primary, secondary):
    denom = primary + secondary
    ai = np.zeros_like(primary)
    np.divide(primary - secondary, denom, out=ai, where=denom != 0)
    return 100.0 * ai


def _log_ratio(primary, secondary):
    return np.log(primary + _EPSILON) - np.log(secondary + _EPSILON)


def _temporal_abs_diff(values):
    return np.abs(np.diff(values, axis=1, prepend=values[:, :1]))


def elaborate_magnitude(operation_type, D, ND):
    """
    D, ND: shape (n_windows, WINDOW_SIZE, 3)

    Available asymmetry-oriented enrichments:
    - raw_enmo: raw axes + wrist ENMO
    - raw_ai: raw axes + magnitude asymmetry index
    - raw_ratio: raw axes + log ENMO ratio
    - raw_jerk: raw axes + magnitude jerk
    - raw_enmo_ai_ratio_jerk: raw axes + ENMO + asymmetry index + log ENMO ratio + jerk
    - enmo_asymmetry: ENMO-only bilateral/asymmetry channels
    """

    if operation_type in _MAGNITUDE_TYPES:
        mag_D = np.linalg.norm(D, axis=2)
        mag_ND = np.linalg.norm(ND, axis=2)

    if operation_type in {
        "ai",
        "raw_enmo_ai",
        "raw_ai",
        "raw_enmo_ai_ratio_jerk",
        "enmo_asymmetry",
    }:
        ai = _asymmetry_index(mag_D, mag_ND)

    if operation_type in {
        "enmo",
        "raw_enmo_ai",
        "raw_enmo",
        "raw_ratio",
        "raw_enmo_ai_ratio_jerk",
        "enmo_asymmetry",
    }:
        enmo_D = np.maximum(mag_D - 1.0, 0.0)
        enmo_ND = np.maximum(mag_ND - 1.0, 0.0)

    if operation_type in {"raw_ratio", "raw_enmo_ai_ratio_jerk", "enmo_asymmetry"}:
        enmo_log_ratio = _log_ratio(enmo_D, enmo_ND)

    if operation_type in {"raw_jerk", "raw_enmo_ai_ratio_jerk"}:
        jerk_D = _temporal_abs_diff(mag_D)
        jerk_ND = _temporal_abs_diff(mag_ND)

    if operation_type == "enmo_asymmetry":
        bilateral_enmo = enmo_D + enmo_ND
        enmo_abs_diff = np.abs(enmo_D - enmo_ND)

    if operation_type == "concat":
        return np.concatenate((mag_D, mag_ND), axis=1)

    elif operation_type == "difference":
        return mag_D - mag_ND

    elif operation_type == "ai":
        return ai

    # ===== MAGNITUDE =====
    elif operation_type == "magnitude":
        return np.stack((mag_D, mag_ND), axis=2)
        # shape: (n_windows, WINDOW_SIZE, 2)

    # ===== ENMO =====
    elif operation_type == "enmo":
        return np.stack((enmo_D, enmo_ND), axis=2)
        # shape: (n_windows, WINDOW_SIZE, 2)

    # ===== RAW =====
    elif operation_type == "raw":
        return np.concatenate((D, ND), axis=2)
        # shape: (n_windows, WINDOW_SIZE, 6)

    # ===== RAW + ENMO =====
    elif operation_type == "raw_enmo":
        return np.concatenate(
            (
                D,
                ND,
                enmo_D[..., None],
                enmo_ND[..., None],
            ),
            axis=2,
        )
        # shape: (n_windows, WINDOW_SIZE, 8)

    # ===== RAW + ASYMMETRY INDEX =====
    elif operation_type == "raw_ai":
        return np.concatenate(
            (
                D,
                ND,
                ai[..., None],
            ),
            axis=2,
        )
        # shape: (n_windows, WINDOW_SIZE, 7)

    # ===== RAW + LOG ENMO RATIO =====
    elif operation_type == "raw_ratio":
        return np.concatenate(
            (
                D,
                ND,
                enmo_log_ratio[..., None],
            ),
            axis=2,
        )
        # shape: (n_windows, WINDOW_SIZE, 7)

    # ===== RAW + MAGNITUDE JERK =====
    elif operation_type == "raw_jerk":
        return np.concatenate(
            (
                D,
                ND,
                jerk_D[..., None],
                jerk_ND[..., None],
            ),
            axis=2,
        )
        # shape: (n_windows, WINDOW_SIZE, 8)

    # ===== RAW + ENMO + AI + LOG ENMO RATIO + MAGNITUDE JERK =====
    elif operation_type == "raw_enmo_ai_ratio_jerk":
        return np.concatenate(
            (
                D,
                ND,
                enmo_D[..., None],
                enmo_ND[..., None],
                ai[..., None],
                enmo_log_ratio[..., None],
                jerk_D[..., None],
                jerk_ND[..., None],
            ),
            axis=2,
        )
        # shape: (n_windows, WINDOW_SIZE, 12)

    # ===== RAW + ENMO + AI =====
    elif operation_type == "raw_enmo_ai":
        return np.concatenate(
            (
                D,
                ND,
                enmo_D[..., None],
                enmo_ND[..., None],
                ai[..., None],
            ),
            axis=2,
        )
        # shape: (n_windows, WINDOW_SIZE, 9)

    # ===== ENMO + BILATERAL / ASYMMETRY CHANNELS =====
    elif operation_type == "enmo_asymmetry":
        return np.stack(
            (
                enmo_D,
                enmo_ND,
                bilateral_enmo,
                enmo_abs_diff,
                enmo_log_ratio,
                ai,
            ),
            axis=2,
        )
        # shape: (n_windows, WINDOW_SIZE, 6)

    else:
        raise ValueError(f"operation type non supportata: {operation_type}")

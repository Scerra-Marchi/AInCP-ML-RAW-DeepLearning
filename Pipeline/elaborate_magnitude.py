import numpy as np

_EPSILON = 1e-6
_MAGNITUDE_TYPES = {
    "concat",
    "difference",
    "ai",
    "magnitude",
    "enmo",
    "enmo_ai",
    "enmo_jerk",
    "raw_enmo_ai",
    "raw_enmo",
    "raw_ai",
    "raw_ratio",
    "raw_jerk",
    "raw_enmo_ai_ratio_jerk",
    "enmo_asymmetry",
    "enmo_asymmetry_jerk",
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
    - enmo_ai: ENMO per wrist + ENMO-based asymmetry index
    - enmo_jerk: ENMO per wrist + ENMO jerk per wrist
    - raw_enmo: raw axes + wrist ENMO
    - raw_ai: raw axes + ENMO-based asymmetry index
    - raw_ratio: raw axes + log ENMO ratio
    - raw_jerk: raw axes + ENMO jerk
    - raw_enmo_ai_ratio_jerk: raw axes + ENMO + asymmetry index + log ENMO ratio + ENMO jerk
    - enmo_asymmetry: ENMO-only bilateral/asymmetry channels
    - enmo_asymmetry_jerk: ENMO-only bilateral/asymmetry channels + jerk
    """

    if operation_type in _MAGNITUDE_TYPES:
        mag_D = np.linalg.norm(D, axis=2)
        mag_ND = np.linalg.norm(ND, axis=2)

    if operation_type in {
        "ai",
        "enmo",
        "enmo_ai",
        "enmo_jerk",
        "raw_enmo_ai",
        "raw_enmo",
        "raw_ai",
        "raw_ratio",
        "raw_enmo_ai_ratio_jerk",
        "enmo_asymmetry",
        "enmo_asymmetry_jerk",
    }:
        enmo_D = np.maximum(mag_D - 1.0, 0.0)
        enmo_ND = np.maximum(mag_ND - 1.0, 0.0)

    if operation_type in {
        "ai",
        "enmo_ai",
        "raw_enmo_ai",
        "raw_ai",
        "raw_enmo_ai_ratio_jerk",
        "enmo_asymmetry",
        "enmo_asymmetry_jerk",
    }:
        ai = _asymmetry_index(enmo_D, enmo_ND)

    if operation_type in {"raw_ratio", "raw_enmo_ai_ratio_jerk", "enmo_asymmetry", "enmo_asymmetry_jerk"}:
        enmo_log_ratio = _log_ratio(enmo_D, enmo_ND)

    if operation_type in {"enmo_jerk", "raw_jerk", "raw_enmo_ai_ratio_jerk", "enmo_asymmetry_jerk"}:
        jerk_D = _temporal_abs_diff(enmo_D)
        jerk_ND = _temporal_abs_diff(enmo_ND)

    if operation_type in {"enmo_asymmetry", "enmo_asymmetry_jerk"}:
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

    # ===== ENMO + AI =====
    elif operation_type == "enmo_ai":
        return np.stack((enmo_D, enmo_ND, ai), axis=2)
        # shape: (n_windows, WINDOW_SIZE, 3)

    # ===== ENMO + JERK =====
    elif operation_type == "enmo_jerk":
        return np.stack((enmo_D, enmo_ND, jerk_D, jerk_ND), axis=2)
        # shape: (n_windows, WINDOW_SIZE, 4)

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

    # ===== RAW + ENMO JERK =====
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

    # ===== RAW + ENMO + AI + LOG ENMO RATIO + ENMO JERK =====
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

    # ===== ENMO + BILATERAL / ASYMMETRY CHANNELS + ENMO JERK =====
    elif operation_type == "enmo_asymmetry_jerk":
        return np.stack(
            (
                enmo_D,
                enmo_ND,
                bilateral_enmo,
                enmo_abs_diff,
                enmo_log_ratio,
                ai,
                jerk_D,
                jerk_ND,
            ),
            axis=2,
        )
        # shape: (n_windows, WINDOW_SIZE, 8)

    else:
        raise ValueError(f"operation type non supportata: {operation_type}")

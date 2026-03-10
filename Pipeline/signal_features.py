import numpy as np

DEFAULT_SIGNAL_EPS = 1e-6
DEFAULT_WINDOW_STD_TOL = 0.005
_FEATURE_BUILDER_TYPES = {
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


def compute_magnitude(values):
    return np.linalg.norm(values, axis=2)


def compute_enmo(magnitude):
    return np.maximum(magnitude - 1.0, 0.0)


def compute_asymmetry_index(primary, secondary):
    denom = primary + secondary
    ai = np.zeros_like(primary)
    np.divide(primary - secondary, denom, out=ai, where=denom != 0)
    return 100.0 * ai


def compute_log_ratio(primary, secondary, epsilon=DEFAULT_SIGNAL_EPS):
    return np.log(primary + epsilon) - np.log(secondary + epsilon)


def compute_temporal_abs_diff(values):
    return np.abs(np.diff(values, axis=1, prepend=values[:, :1]))


def compute_invalid_bitmap(D_w, ND_w, std_tol=DEFAULT_WINDOW_STD_TOL):
    combined = np.concatenate((D_w, ND_w), axis=2)
    std_features = np.std(combined, axis=1)
    return np.all(std_features < std_tol, axis=1)


def compute_window_sensor_features(
    D_w,
    ND_w,
    *,
    epsilon=DEFAULT_SIGNAL_EPS,
    std_tol=DEFAULT_WINDOW_STD_TOL,
):
    mag_d = compute_magnitude(D_w).astype(np.float32)
    mag_nd = compute_magnitude(ND_w).astype(np.float32)
    enmo_d = compute_enmo(mag_d).astype(np.float32)
    enmo_nd = compute_enmo(mag_nd).astype(np.float32)
    jerk_d = compute_temporal_abs_diff(enmo_d).astype(np.float32)
    jerk_nd = compute_temporal_abs_diff(enmo_nd).astype(np.float32)

    enmo_mean_d = enmo_d.mean(axis=1).astype(np.float32)
    enmo_mean_nd = enmo_nd.mean(axis=1).astype(np.float32)
    enmo_diff = (enmo_mean_d - enmo_mean_nd).astype(np.float32)
    bilateral_enmo_mean = (enmo_mean_d + enmo_mean_nd).astype(np.float32)

    features = {
        "enmo_mean_d": enmo_mean_d,
        "enmo_mean_nd": enmo_mean_nd,
        "enmo_diff": enmo_diff,
        "enmo_log_ratio": compute_log_ratio(enmo_mean_d, enmo_mean_nd, epsilon=epsilon).astype(np.float32),
        "signed_ai_enmo": np.divide(
            enmo_diff,
            bilateral_enmo_mean + epsilon,
        ).astype(np.float32),
        "bilateral_enmo_mean": bilateral_enmo_mean,
        "jerk_mean_d": jerk_d.mean(axis=1).astype(np.float32),
        "jerk_mean_nd": jerk_nd.mean(axis=1).astype(np.float32),
    }
    invalid_bitmap = compute_invalid_bitmap(D_w, ND_w, std_tol=std_tol).astype(np.uint8)
    return features, invalid_bitmap


def build_signal_features(operation_type, D, ND):
    """
    D, ND: shape (n_windows, WINDOW_SIZE, 3)
    """

    if operation_type in _FEATURE_BUILDER_TYPES:
        mag_D = compute_magnitude(D)
        mag_ND = compute_magnitude(ND)

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
        enmo_D = compute_enmo(mag_D)
        enmo_ND = compute_enmo(mag_ND)

    if operation_type in {
        "ai",
        "enmo_ai",
        "raw_enmo_ai",
        "raw_ai",
        "raw_enmo_ai_ratio_jerk",
        "enmo_asymmetry",
        "enmo_asymmetry_jerk",
    }:
        ai = compute_asymmetry_index(enmo_D, enmo_ND)

    if operation_type in {"raw_ratio", "raw_enmo_ai_ratio_jerk", "enmo_asymmetry", "enmo_asymmetry_jerk"}:
        enmo_log_ratio = compute_log_ratio(enmo_D, enmo_ND)

    if operation_type in {"enmo_jerk", "raw_jerk", "raw_enmo_ai_ratio_jerk", "enmo_asymmetry_jerk"}:
        jerk_D = compute_temporal_abs_diff(enmo_D)
        jerk_ND = compute_temporal_abs_diff(enmo_ND)

    if operation_type in {"enmo_asymmetry", "enmo_asymmetry_jerk"}:
        bilateral_enmo = enmo_D + enmo_ND
        enmo_abs_diff = np.abs(enmo_D - enmo_ND)

    if operation_type == "concat":
        return np.concatenate((mag_D, mag_ND), axis=1)

    if operation_type == "difference":
        return mag_D - mag_ND

    if operation_type == "ai":
        return ai

    if operation_type == "magnitude":
        return np.stack((mag_D, mag_ND), axis=2)

    if operation_type == "enmo":
        return np.stack((enmo_D, enmo_ND), axis=2)

    if operation_type == "enmo_ai":
        return np.stack((enmo_D, enmo_ND, ai), axis=2)

    if operation_type == "enmo_jerk":
        return np.stack((enmo_D, enmo_ND, jerk_D, jerk_ND), axis=2)

    if operation_type == "raw":
        return np.concatenate((D, ND), axis=2)

    if operation_type == "raw_enmo":
        return np.concatenate((D, ND, enmo_D[..., None], enmo_ND[..., None]), axis=2)

    if operation_type == "raw_ai":
        return np.concatenate((D, ND, ai[..., None]), axis=2)

    if operation_type == "raw_ratio":
        return np.concatenate((D, ND, enmo_log_ratio[..., None]), axis=2)

    if operation_type == "raw_jerk":
        return np.concatenate((D, ND, jerk_D[..., None], jerk_ND[..., None]), axis=2)

    if operation_type == "raw_enmo_ai_ratio_jerk":
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

    if operation_type == "raw_enmo_ai":
        return np.concatenate((D, ND, enmo_D[..., None], enmo_ND[..., None], ai[..., None]), axis=2)

    if operation_type == "enmo_asymmetry":
        return np.stack((enmo_D, enmo_ND, bilateral_enmo, enmo_abs_diff, enmo_log_ratio, ai), axis=2)

    if operation_type == "enmo_asymmetry_jerk":
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

    raise ValueError(f"Unsupported operation_type: {operation_type}")

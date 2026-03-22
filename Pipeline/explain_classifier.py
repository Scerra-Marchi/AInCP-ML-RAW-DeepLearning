import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from torch import nn

from create_windows import create_windows
from plotting import (
    _apply_axis_style,
    _configure_plot_style,
    _plot_global_feature_importance,
    _plot_global_shap_summary,
    _save_figure,
)
from predict_samples import build_estimators_list


CLASSIFIER_DISPLAY_NAME = "Best classifier (Classifier 1)"
CLASSIFIER_FILE_STEM = "best_classifier_1"


class _BinaryClassifierShapWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        if out.ndim == 1:
            return out.unsqueeze(-1)
        return out


def _classifier_feature_names(method: str, n_features: int) -> list[str]:
    feature_names_by_method = {
        "concat": ["Dominant + non-dominant magnitude"],
        "difference": ["Dominant - non-dominant magnitude"],
        "ai": ["Asymmetry index"],
        "magnitude": ["Dominant magnitude", "Non-dominant magnitude"],
        "enmo": ["Dominant ENMO", "Non-dominant ENMO"],
        "enmo_ai": ["Dominant ENMO", "Non-dominant ENMO", "Asymmetry index"],
        "enmo_ai_jerk": [
            "Dominant ENMO",
            "Non-dominant ENMO",
            "Asymmetry index",
            "Dominant jerk",
            "Non-dominant jerk",
        ],
        "enmo_jerk": [
            "Dominant ENMO",
            "Non-dominant ENMO",
            "Dominant jerk",
            "Non-dominant jerk",
        ],
        "raw": [
            "Dominant X",
            "Dominant Y",
            "Dominant Z",
            "Non-dominant X",
            "Non-dominant Y",
            "Non-dominant Z",
        ],
        "raw_enmo": [
            "Dominant X",
            "Dominant Y",
            "Dominant Z",
            "Non-dominant X",
            "Non-dominant Y",
            "Non-dominant Z",
            "Dominant ENMO",
            "Non-dominant ENMO",
        ],
        "raw_ai": [
            "Dominant X",
            "Dominant Y",
            "Dominant Z",
            "Non-dominant X",
            "Non-dominant Y",
            "Non-dominant Z",
            "Asymmetry index",
        ],
        "raw_ratio": [
            "Dominant X",
            "Dominant Y",
            "Dominant Z",
            "Non-dominant X",
            "Non-dominant Y",
            "Non-dominant Z",
            "ENMO log-ratio",
        ],
        "raw_jerk": [
            "Dominant X",
            "Dominant Y",
            "Dominant Z",
            "Non-dominant X",
            "Non-dominant Y",
            "Non-dominant Z",
            "Dominant jerk",
            "Non-dominant jerk",
        ],
        "raw_enmo_ai_ratio_jerk": [
            "Dominant X",
            "Dominant Y",
            "Dominant Z",
            "Non-dominant X",
            "Non-dominant Y",
            "Non-dominant Z",
            "Dominant ENMO",
            "Non-dominant ENMO",
            "Asymmetry index",
            "ENMO log-ratio",
            "Dominant jerk",
            "Non-dominant jerk",
        ],
        "raw_enmo_ai": [
            "Dominant X",
            "Dominant Y",
            "Dominant Z",
            "Non-dominant X",
            "Non-dominant Y",
            "Non-dominant Z",
            "Dominant ENMO",
            "Non-dominant ENMO",
            "Asymmetry index",
        ],
        "raw_enmo_ai_jerk": [
            "Dominant X",
            "Dominant Y",
            "Dominant Z",
            "Non-dominant X",
            "Non-dominant Y",
            "Non-dominant Z",
            "Dominant ENMO",
            "Non-dominant ENMO",
            "Asymmetry index",
            "Dominant jerk",
            "Non-dominant jerk",
        ],
        "enmo_asymmetry": [
            "Dominant ENMO",
            "Non-dominant ENMO",
            "Bilateral ENMO",
            "ENMO absolute difference",
            "ENMO log-ratio",
            "Asymmetry index",
        ],
        "enmo_asymmetry_jerk": [
            "Dominant ENMO",
            "Non-dominant ENMO",
            "Bilateral ENMO",
            "ENMO absolute difference",
            "ENMO log-ratio",
            "Asymmetry index",
            "Dominant jerk",
            "Non-dominant jerk",
        ],
    }
    feature_names = feature_names_by_method.get(method)
    if feature_names is None or len(feature_names) != n_features:
        return [f"Feature {idx + 1}" for idx in range(n_features)]
    return feature_names


def _align_attr_to_inputs(attr: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    attr = np.asarray(attr)
    inputs = np.asarray(inputs, dtype=float)

    if inputs.ndim == 3:
        attr = np.squeeze(attr)
        if attr.ndim == 2:
            attr = attr[..., None]
        if attr.shape != inputs.shape:
            attr = attr.reshape(inputs.shape)
        return attr

    if inputs.ndim == 2:
        attr = np.squeeze(attr)
        if attr.ndim == 3 and attr.shape[-1] == 1:
            attr = attr[..., 0]
        if attr.ndim == 1:
            attr = attr.reshape(inputs.shape[0], -1)
        if attr.shape != inputs.shape:
            attr = attr.reshape(inputs.shape)
        return attr

    raise ValueError(f"Unsupported classifier SHAP input shape: {inputs.shape}.")


def _collapse_over_time(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:
        return values.mean(axis=1)
    if values.ndim == 2:
        return values
    raise ValueError(f"Unsupported classifier matrix shape: {values.shape}.")


def _plot_signed_feature_importance(stats_folder, feature_names, signed_importance):
    signed_importance = np.asarray(signed_importance, dtype=float)
    order = np.argsort(np.abs(signed_importance))[::-1]
    ordered_names = [feature_names[idx] for idx in order]
    ordered_importance = signed_importance[order]
    colors = ["#c44e52" if value >= 0 else "#4c72b0" for value in ordered_importance[::-1]]

    fig_height = max(4.4, 1.8 + 0.24 * len(ordered_names))
    fig, ax = plt.subplots(figsize=(6.6, fig_height), constrained_layout=True)
    ax.barh(np.arange(len(ordered_names)), ordered_importance[::-1], color=colors)
    ax.set_yticks(np.arange(len(ordered_names)))
    ax.set_yticklabels(ordered_names[::-1])
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean signed SHAP value")
    ax.set_title(f"Signed feature importance for {CLASSIFIER_DISPLAY_NAME}")
    _apply_axis_style(ax, grid_axis="x")
    _save_figure(fig, os.path.join(stats_folder, f"{CLASSIFIER_FILE_STEM}_signed_feature_importance"))


def _batch_feature_sums(attr: np.ndarray, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    inputs = np.asarray(inputs, dtype=float)
    attr = _align_attr_to_inputs(attr, inputs)

    if inputs.ndim == 3:
        return (
            np.abs(attr).sum(axis=(0, 1)),
            inputs.sum(axis=(0, 1)),
            int(inputs.shape[0] * inputs.shape[1]),
        )

    if inputs.ndim == 2:
        return (
            np.array([float(np.abs(attr).sum())], dtype=float),
            np.array([float(inputs.sum())], dtype=float),
            int(inputs.size),
        )

    raise ValueError(f"Unsupported classifier SHAP input shape: {inputs.shape}.")


def explain_best_classifier(
    data_folder,
    save_folder,
    metadata,
    window_size=None,
    decimation_factor=1,
):
    _configure_plot_style()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    best_estimators_df = pd.read_csv(
        save_folder + "best_estimators_results.csv",
        index_col=0,
    ).sort_values(by=["mean_test_score", "std_test_score"], ascending=False)

    if window_size is not None:
        best_estimators_df = best_estimators_df[best_estimators_df["window_size"] == window_size]
    if decimation_factor is not None:
        best_estimators_df = best_estimators_df[
            best_estimators_df["decimation_factor"] == decimation_factor
        ]
    if best_estimators_df.empty:
        raise ValueError("No classifier is available for explainability with the requested filters.")

    best_only_df = best_estimators_df.iloc[[0]]
    estimators_specs_list, estimators_list = build_estimators_list(
        best_estimators_df=best_only_df,
        save_folder=save_folder,
        min_mean_test_score=-np.inf,
        window_size=int(best_only_df.iloc[0]["window_size"]),
        decimation_factor=int(best_only_df.iloc[0]["decimation_factor"]),
    )
    if not estimators_list:
        raise ValueError("The best classifier could not be loaded for explainability.")

    best_specs = estimators_specs_list[0]
    estimator = estimators_list[0]["estimator"]
    method = str(best_specs["method"])
    selected_window_size = int(best_specs["window_size"])
    selected_decimation_factor = int(best_specs["decimation_factor"])

    stats_folder = os.path.join(save_folder, "Classifier_explainability")
    os.makedirs(stats_folder, exist_ok=True)

    X, y_aha, _, y_binary, invalid_bitmap, subjects = create_windows(
        data_folder=data_folder,
        metadata=metadata,
        operation_type=method,
        WINDOW_SIZE=selected_window_size,
        decimation_factor=selected_decimation_factor,
        input_type="AHA",
    )
    X = np.asarray(X)
    invalid_mask = np.asarray(invalid_bitmap, dtype=bool)
    valid_mask = ~invalid_mask

    if not np.any(valid_mask):
        raise ValueError("No valid AHA windows are available for classifier explainability.")

    X_valid = X[valid_mask]
    y_binary_valid = np.asarray(y_binary, dtype=np.int64)[valid_mask]
    y_aha_valid = np.asarray(y_aha, dtype=float)[valid_mask]
    subjects_valid = np.asarray(subjects, dtype=int)[valid_mask]

    scaler = estimator.named_steps["scaler"]
    classifier = estimator.named_steps["model"]
    net = classifier.module_
    net.eval()
    shap_model = _BinaryClassifierShapWrapper(net).to(next(net.parameters()).device)
    shap_model.eval()

    X_valid_scaled = scaler.transform(X_valid).astype(np.float32)
    predicted_probability = estimator.predict_proba(X_valid)[:, 1]

    feature_count = X_valid_scaled.shape[-1] if X_valid_scaled.ndim == 3 else 1
    feature_names = _classifier_feature_names(method, feature_count)

    device = next(net.parameters()).device
    background_scaled = X_valid_scaled
    background_tensor = torch.tensor(background_scaled, dtype=torch.float32).to(device)
    explain_tensor = background_tensor

    cudnn_enabled = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        explainer = shap.GradientExplainer(shap_model, background_tensor)
        shap_values = explainer.shap_values(explain_tensor)
    finally:
        torch.backends.cudnn.enabled = cudnn_enabled

    attr = shap_values[0] if isinstance(shap_values, list) else shap_values
    attr = _align_attr_to_inputs(attr, X_valid_scaled)
    shap_abs_sum, feature_sum, value_count = _batch_feature_sums(attr, X_valid_scaled)

    collapsed_attr = _collapse_over_time(attr)
    collapsed_feature_values = _collapse_over_time(X_valid_scaled)
    signed_feature_importance = np.nanmean(collapsed_attr, axis=0)

    global_feature_importance = shap_abs_sum / max(value_count, 1)
    mean_feature_value = feature_sum / max(value_count, 1)

    feature_importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap_value": global_feature_importance,
            "mean_signed_shap_value": signed_feature_importance,
            "mean_feature_value_test": mean_feature_value,
        }
    ).sort_values(by="mean_abs_shap_value", ascending=False, ignore_index=True)
    feature_importance_df.to_csv(
        os.path.join(stats_folder, f"{CLASSIFIER_FILE_STEM}_global_feature_importance.csv"),
        index=False,
    )

    _plot_global_feature_importance(
        stats_folder,
        feature_importance_df["feature"].tolist(),
        [feature_importance_df["mean_abs_shap_value"].to_numpy(dtype=float)],
        title=f"Feature importance for {CLASSIFIER_DISPLAY_NAME}",
        file_stem=f"{CLASSIFIER_FILE_STEM}_global_feature_importance",
    )
    _plot_signed_feature_importance(
        stats_folder,
        feature_importance_df["feature"].tolist(),
        feature_importance_df["mean_signed_shap_value"].to_numpy(dtype=float),
    )
    _plot_global_shap_summary(
        stats_folder,
        [collapsed_attr],
        [collapsed_feature_values],
        feature_names,
        title=f"SHAP summary for {CLASSIFIER_DISPLAY_NAME}",
        file_stem=f"{CLASSIFIER_FILE_STEM}_shap_summary",
    )

    explainability_stats = {
        "Classifier Label": CLASSIFIER_DISPLAY_NAME,
        "Method": method,
        "Window Size": selected_window_size,
        "Decimation Factor": selected_decimation_factor,
        "Model Type": str(best_specs["model_type"]),
        "Gridsearch Hash": str(best_specs["gridsearch_hash"]),
        "Mean Test Score": float(best_only_df.iloc[0]["mean_test_score"]),
        "Std Test Score": float(best_only_df.iloc[0]["std_test_score"]),
        "Total AHA Windows": int(X.shape[0]),
        "Valid AHA Windows": int(X_valid.shape[0]),
        "Test Subjects": sorted(np.unique(subjects_valid).astype(int).tolist()),
        "Healthy Window Fraction": float(np.mean(y_binary_valid == 1)),
        "Mean AHA Across Valid Windows": float(np.mean(y_aha_valid)),
        "Mean Predicted Probability": float(np.mean(predicted_probability)),
        "Background Samples": int(background_scaled.shape[0]),
        "Top Features": feature_importance_df.head(10).to_dict(orient="records"),
    }
    with open(os.path.join(stats_folder, f"{CLASSIFIER_FILE_STEM}_explainability_stats.json"), "w") as file:
        json.dump(explainability_stats, file, indent=4)

    print(f"Selected {CLASSIFIER_DISPLAY_NAME} for explainability:")
    print(
        f" - {best_specs['model_type']} | method={method} | "
        f"window_size={selected_window_size} | decimation_factor={selected_decimation_factor}"
    )
    print("Top global classifier SHAP features:")
    for _, row in feature_importance_df.head(10).iterrows():
        print(f" - {row['feature']}: {row['mean_abs_shap_value']:.6f}")

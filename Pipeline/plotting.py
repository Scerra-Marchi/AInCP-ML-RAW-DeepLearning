import os
import json
import warnings
import pandas as pd
from itertools import product
import joblib as jl
import numpy as np
from predict_samples import build_estimators_list, predict_samples
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection
from train_regressor import (
    build_block_feature_names,
    build_regressor_sample,
    regressor_model_path,
)
from signal_features import compute_enmo
from read_file import read_file

import torch
import shap

THESIS_PDF_DPI = 300
THESIS_PLOT_SIZES = {
    "timeline": (6.4, 2.8),
    "summary": (6.4, 4.2),
    "heatmap": (6.6, 4.0),
    "wide": (6.8, 3.4),
    "corrcoeff": (9.0, 3.4),
}
DAILY_INDICATOR_EPS = 1e-6
_PLOT_STYLE_CONFIGURED = False


def _configure_plot_style():
    global _PLOT_STYLE_CONFIGURED
    if _PLOT_STYLE_CONFIGURED:
        return

    base_params = {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "axes.unicode_minus": False,
    }
    matplotlib.rcParams.update(base_params)
    matplotlib.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        }
    )

    _PLOT_STYLE_CONFIGURED = True


def _figure_size(kind):
    return THESIS_PLOT_SIZES[kind]


def _day_number_label(day_value, first_day_value):
    return f"Day {int(round(float(day_value) - float(first_day_value))) + 1}"


def _slot_hour_edges(slot_hours):
    slot_hours = np.asarray(slot_hours, dtype=float)
    if slot_hours.size == 0:
        return np.array([0.0, 24.0], dtype=float)
    if slot_hours.size == 1:
        half_width = 0.5
        edges = np.array([slot_hours[0] - half_width, slot_hours[0] + half_width], dtype=float)
    else:
        diffs = np.diff(slot_hours)
        edges = np.empty(slot_hours.size + 1, dtype=float)
        edges[1:-1] = 0.5 * (slot_hours[:-1] + slot_hours[1:])
        edges[0] = slot_hours[0] - 0.5 * diffs[0]
        edges[-1] = slot_hours[-1] + 0.5 * diffs[-1]
    return np.clip(edges, 0.0, 24.0)


def _time_edges(x_values):
    x_values = np.asarray(x_values, dtype=float)
    if x_values.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    if x_values.size == 1:
        half_width = 0.5 / 24.0
        return np.array([x_values[0] - half_width, x_values[0] + half_width], dtype=float)
    diffs = np.diff(x_values)
    edges = np.empty(x_values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (x_values[:-1] + x_values[1:])
    edges[0] = x_values[0] - 0.5 * diffs[0]
    edges[-1] = x_values[-1] + 0.5 * diffs[-1]
    return edges


def _expand_regular_grid(values, *arrays, axis=-1, period=None):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return values, arrays

    diffs = np.diff(values)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return values, arrays

    step = float(np.median(diffs))
    if period is None:
        grid = values[0] + np.arange(int(round((values[-1] - values[0]) / step)) + 1, dtype=float) * step
        indexes = np.rint((values - values[0]) / step).astype(int)
    else:
        phase = float(np.mod(values[0], step))
        grid = phase + np.arange(int(np.floor((period - phase - 1e-12) / step)) + 1, dtype=float) * step
        indexes = np.rint((values - grid[0]) / step).astype(int)

    if np.any(indexes < 0) or np.any(indexes >= grid.size):
        return values, arrays
    if np.max(np.abs(grid[indexes] - values)) > 0.25 * step:
        return values, arrays

    expanded_arrays = []
    for arr in arrays:
        arr_np = np.asarray(arr, dtype=float)
        target_shape = list(arr_np.shape)
        target_shape[axis] = grid.size
        expanded = np.full(target_shape, np.nan, dtype=float)
        target = [slice(None)] * expanded.ndim
        source = [slice(None)] * arr_np.ndim
        for source_idx, target_idx in enumerate(indexes):
            target[axis] = target_idx
            source[axis] = source_idx
            expanded[tuple(target)] = arr_np[tuple(source)]
        expanded_arrays.append(expanded)

    return grid, tuple(expanded_arrays)


def _set_real_time_heatmap_ticks(ax, timeline_timestamps):
    timeline_timestamps = np.asarray(timeline_timestamps, dtype=float)
    if timeline_timestamps.size == 0:
        return

    spans_multiple_days = np.floor(timeline_timestamps[-1]) > np.floor(timeline_timestamps[0])
    if spans_multiple_days:
        day_numbers = np.floor(timeline_timestamps)
        unique_days, first_positions = np.unique(day_numbers, return_index=True)
        tick_positions = timeline_timestamps[first_positions]
        first_day = unique_days[0]
        tick_labels = [_day_number_label(day_value, first_day) for day_value in unique_days]
    else:
        tick_count = min(6, timeline_timestamps.size)
        tick_indexes = np.linspace(0, timeline_timestamps.size - 1, tick_count, dtype=int)
        tick_indexes = np.unique(tick_indexes)
        tick_positions = timeline_timestamps[tick_indexes]
        tick_labels = [matplotlib.dates.num2date(float(tick)).strftime("%H:%M") for tick in tick_positions]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)


def _new_figure(kind, *, nrows=1, ncols=1, sharex=False, gridspec_kw=None):
    return plt.subplots(
        nrows,
        ncols,
        figsize=_figure_size(kind),
        sharex=sharex,
        gridspec_kw=gridspec_kw,
        constrained_layout=True,
    )


def _format_time_axis(ax):
    locator = matplotlib.dates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))


def _apply_axis_style(ax, *, grid_axis="y"):
    if grid_axis:
        ax.grid(axis=grid_axis, alpha=0.25)


def _save_figure(fig, path_base):
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.03}
    fig.savefig(path_base + ".pdf", dpi=THESIS_PDF_DPI, **save_kwargs)
    plt.close(fig)


def _plot_value_colored_line(
    ax,
    x_values,
    y_values,
    *,
    cmap,
    vmin,
    vmax,
    linewidth=2.0,
    outline_color="#222222",
    outline_linewidth=None,
):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
    if not np.any(valid_mask):
        return

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    valid_indexes = np.flatnonzero(valid_mask)
    split_points = np.where(np.diff(valid_indexes) > 1)[0] + 1
    segments_indexes = np.split(valid_indexes, split_points)
    outline_width = linewidth + 1.4 if outline_linewidth is None else outline_linewidth

    for segment_indexes in segments_indexes:
        x_segment = x_values[segment_indexes]
        y_segment = y_values[segment_indexes]
        if outline_color is not None:
            if x_segment.size == 1:
                ax.scatter(x_segment, y_segment, color=outline_color, s=10, zorder=2, edgecolors="none")
            else:
                ax.plot(
                    x_segment,
                    y_segment,
                    color=outline_color,
                    linewidth=outline_width,
                    solid_capstyle="round",
                    zorder=2,
                )
                ax.scatter(x_segment, y_segment, color=outline_color, s=8, zorder=2, edgecolors="none")

        if x_segment.size == 1:
            ax.scatter(x_segment, y_segment, c=y_segment, cmap=cmap, norm=norm, s=6, zorder=3, edgecolors="none")
            continue

        points = np.column_stack((x_segment, y_segment)).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        segment_values = 0.5 * (y_segment[:-1] + y_segment[1:])
        line_collection = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidths=linewidth,
            zorder=3,
        )
        line_collection.set_array(segment_values)
        ax.add_collection(line_collection)
        ax.scatter(x_segment, y_segment, c=y_segment, cmap=cmap, norm=norm, s=4, zorder=4, edgecolors="none")

    ax.autoscale_view()


def _add_formula_label(ax, formula_text):
    ax.text(
        0.985,
        0.82,
        formula_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )


def _percent_label(text):
    if matplotlib.rcParams.get("text.usetex", False):
        return text.replace("%", r"\%")
    return text


def _transform_regressor_input_for_shap(regressor, model_input):
    prep = regressor.named_steps["prep"]
    scaler = regressor.named_steps["scaler"]
    regressor_sequence = prep.transform(model_input)
    regressor_sequence_scaled = scaler.transform(regressor_sequence)
    return regressor_sequence, regressor_sequence_scaled


def _enable_regressor_shap_output(net):
    previous_return_last_step = getattr(net, "return_last_step", None)
    previous_keep_output_dim = getattr(net, "keep_output_dim", None)

    if previous_return_last_step is not None:
        net.return_last_step = True
    if previous_keep_output_dim is not None:
        net.keep_output_dim = True

    return previous_return_last_step, previous_keep_output_dim


def _restore_regressor_output_flags(net, previous_return_last_step, previous_keep_output_dim):
    if previous_return_last_step is not None:
        net.return_last_step = previous_return_last_step
    if previous_keep_output_dim is not None:
        net.keep_output_dim = previous_keep_output_dim


def _build_regressor_shap_explainer(
    *,
    regressor,
    background_samples,
    device,
):
    net = regressor.named_steps["model"].module_
    net.eval()

    if not background_samples:
        raise ValueError("plot_dashboards metadata is empty.")

    background_sequence_scaled = np.stack(
        [entry["regressor_sequence_scaled"] for entry in background_samples]
    ).astype(np.float32)
    background_tensor = torch.tensor(background_sequence_scaled, dtype=torch.float32).to(device)

    previous_return_last_step, previous_keep_output_dim = _enable_regressor_shap_output(net)
    try:
        return shap.GradientExplainer(net, background_tensor)
    finally:
        _restore_regressor_output_flags(net, previous_return_last_step, previous_keep_output_dim)


def _timeline_bar_width(timeline_timestamps):
    timeline_timestamps = np.asarray(timeline_timestamps, dtype=float)
    if timeline_timestamps.size < 2:
        return 1.0 / 24.0

    diffs = np.diff(np.sort(timeline_timestamps))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1.0 / 24.0
    return float(0.8 * np.median(diffs))


def _aggregate_window_timeline(
    *,
    prep,
    window_timestamps,
    invalid_mask,
    window_size,
    decimation_factor,
    series=None,
):
    window_timestamps = np.asarray(window_timestamps, dtype=float)
    invalid_mask = np.asarray(invalid_mask, dtype=bool)
    series = None if series is None else np.asarray(series, dtype=float)

    prep_mode = getattr(prep, "mode", "block")
    if prep_mode != "block":
        block_slices = [(idx, idx + 1) for idx in range(invalid_mask.size)]
    else:
        fs = getattr(prep, "fs", 80)
        block_seconds = getattr(prep, "block_seconds", 3600)
        seconds_per_window = window_size * decimation_factor / fs
        steps_per_block = max(1, int(round(block_seconds / seconds_per_window)))
        block_slices = [
            (start, min(invalid_mask.size, start + steps_per_block))
            for start in range(0, invalid_mask.size, steps_per_block)
        ]

    block_timestamps = np.zeros(len(block_slices), dtype=float)
    block_invalid_mask = np.zeros(len(block_slices), dtype=bool)
    block_valid_fraction = np.zeros(len(block_slices), dtype=float)
    block_series = None if series is None else np.full(len(block_slices), np.nan, dtype=float)

    for block_idx, (start, end) in enumerate(block_slices):
        block_timestamps[block_idx] = float(window_timestamps[start:end].mean())
        block_invalid = invalid_mask[start:end]
        valid_mask = ~block_invalid
        block_invalid_mask[block_idx] = bool(np.all(block_invalid))
        block_valid_fraction[block_idx] = float(valid_mask.mean())
        if series is not None and np.any(valid_mask):
            block_series[block_idx] = float(np.mean(series[start:end][valid_mask]))

    return block_timestamps, block_invalid_mask, block_valid_fraction, block_series


def _format_feature_name(name):
    classifier_tokens = name.split("_")
    if len(classifier_tokens) >= 4 and classifier_tokens[0] == "classifier":
        classifier_idx = int(classifier_tokens[1]) + 1
        stat = classifier_tokens[-1]
        stat_labels = {
            "mean": "mean prediction",
            "std": "prediction std",
            "posrate": "positive-rate fraction",
        }
        return f"Classifier {classifier_idx} {stat_labels.get(stat, stat)}"

    feature_labels = {
        "block_enmo_mean_d": "Dominant ENMO mean",
        "block_enmo_mean_nd": "Non-dominant ENMO mean",
        "block_enmo_std_d": "Dominant ENMO std",
        "block_enmo_std_nd": "Non-dominant ENMO std",
        "block_enmo_diff": "ENMO difference",
        "block_enmo_log_ratio": "ENMO log-ratio",
        "block_signed_ai_enmo": "Signed AI ENMO",
        "block_abs_ai_enmo": "Absolute AI ENMO",
        "block_bilateral_enmo_mean": "Bilateral ENMO mean",
        "block_fraction_d_gt_nd": "Dominant > non-dominant fraction",
        "block_jerk_mean_d": "Dominant jerk mean",
        "block_jerk_mean_nd": "Non-dominant jerk mean",
        "block_valid_fraction": "Valid fraction",
        "time_sin": "Time sine",
        "time_cos": "Time cosine",
    }
    return feature_labels.get(name, name.replace("_", " "))


def _feature_group_name(name):
    if name.startswith("classifier_") and name.endswith("_mean"):
        return "Classifier mean prediction"
    if name.startswith("classifier_") and name.endswith("_std"):
        return "Classifier prediction std"
    if name.startswith("classifier_") and name.endswith("_posrate"):
        return "Classifier positive-rate"
    if "enmo" in name:
        return "ENMO features"
    if "jerk" in name:
        return "Jerk features"
    if name == "block_valid_fraction":
        return "Validity"
    if name in {"time_sin", "time_cos"}:
        return "Circadian features"
    return "Other"


def _plot_global_feature_importance(
    global_stats_folder,
    feature_names,
    subject_feature_importances,
    *,
    title="Global feature importance",
    file_stem="global_feature_importance",
):
    if not subject_feature_importances:
        return

    subject_feature_importances = np.asarray(subject_feature_importances, dtype=float)
    global_importance = np.nanmean(subject_feature_importances, axis=0)
    order = np.argsort(global_importance)[::-1]

    ordered_names = [feature_names[idx] for idx in order]
    ordered_importance = global_importance[order]

    fig_height = max(4.4, 1.8 + 0.24 * len(ordered_names))
    fig, ax = plt.subplots(figsize=(6.6, fig_height), constrained_layout=True)
    ax.barh(np.arange(len(ordered_names)), ordered_importance[::-1], color="#2a9d8f")
    ax.set_yticks(np.arange(len(ordered_names)))
    ax.set_yticklabels(ordered_names[::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title)
    _apply_axis_style(ax, grid_axis="x")
    _save_figure(fig, os.path.join(global_stats_folder, file_stem))


def _plot_global_grouped_importance(global_stats_folder, raw_feature_names, subject_feature_importances):
    if not subject_feature_importances:
        return

    group_names = []
    for raw_name in raw_feature_names:
        group_name = _feature_group_name(raw_name)
        if group_name not in group_names:
            group_names.append(group_name)

    grouped_subject_importances = []
    for subject_values in subject_feature_importances:
        grouped_values = []
        subject_values = np.asarray(subject_values, dtype=float)
        for group_name in group_names:
            mask = np.array([_feature_group_name(name) == group_name for name in raw_feature_names], dtype=bool)
            grouped_values.append(float(np.nansum(subject_values[mask])))
        grouped_subject_importances.append(grouped_values)

    grouped_subject_importances = np.asarray(grouped_subject_importances, dtype=float)
    global_grouped_importance = np.nanmean(grouped_subject_importances, axis=0)
    order = np.argsort(global_grouped_importance)[::-1]

    ordered_names = [group_names[idx] for idx in order]
    ordered_importance = global_grouped_importance[order]

    fig_height = max(4.0, 1.8 + 0.4 * len(ordered_names))
    fig, ax = plt.subplots(figsize=(6.4, fig_height), constrained_layout=True)
    ax.barh(np.arange(len(ordered_names)), ordered_importance[::-1], color="#7a6f9b")
    ax.set_yticks(np.arange(len(ordered_names)))
    ax.set_yticklabels(ordered_names[::-1])
    ax.set_xlabel("Mean summed |SHAP value|")
    ax.set_title("Global grouped feature importance")
    _apply_axis_style(ax, grid_axis="x")
    _save_figure(fig, os.path.join(global_stats_folder, "global_grouped_feature_importance"))


def _plot_global_shap_summary(
    global_stats_folder,
    global_attr_rows,
    global_feature_rows,
    feature_names,
    *,
    title="Global SHAP summary",
    file_stem="global_shap_summary",
):
    if not global_attr_rows or not global_feature_rows:
        return

    attr_matrix = np.vstack(global_attr_rows)
    feature_matrix = np.vstack(global_feature_rows)
    summary_height = max(4.4, 1.8 + 0.24 * len(feature_names))

    plt.figure(figsize=(6.4, summary_height))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The NumPy global RNG was seeded by calling `np.random.seed`.*",
            category=FutureWarning,
        )
        shap.summary_plot(
            attr_matrix,
            features=feature_matrix,
            feature_names=feature_names,
            show=False,
            plot_size=None,
            max_display=len(feature_names),
        )
    fig = plt.gcf()
    plt.title(title)
    _save_figure(fig, os.path.join(global_stats_folder, file_stem))


def _plot_global_time_of_day_importance(global_stats_folder, subject_cycle_data_list):
    if not subject_cycle_data_list:
        return

    all_slots = sorted(
        {
            float(slot_value)
            for subject_cycle_data in subject_cycle_data_list
            for slot_value in subject_cycle_data["slot_values"]
        }
    )
    if not all_slots:
        return

    aligned_matrix = np.full((len(subject_cycle_data_list), len(all_slots)), np.nan, dtype=float)
    slot_to_index = {slot_value: idx for idx, slot_value in enumerate(all_slots)}
    for subject_idx, subject_cycle_data in enumerate(subject_cycle_data_list):
        for slot_value, mean_abs_value in zip(
            subject_cycle_data["slot_values"],
            subject_cycle_data["mean_importance"],
        ):
            aligned_matrix[subject_idx, slot_to_index[float(slot_value)]] = float(mean_abs_value)

    global_mean_abs = np.nanmean(aligned_matrix, axis=0)
    if not np.any(np.isfinite(global_mean_abs)):
        return

    slot_hours = np.asarray(all_slots, dtype=float) * 24.0
    fig, ax = _new_figure("timeline")
    ax.plot(slot_hours, global_mean_abs, color="#d17c1f")
    ax.set_xlim(0.0, 24.0)
    ax.set_xlabel("Time of day")
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title("Global time-of-day importance")
    tick_hours = np.arange(0, 24, 2)
    ax.set_xticks(tick_hours)
    ax.set_xticklabels([f"{int(hour):02d}:00" for hour in tick_hours])
    _apply_axis_style(ax, grid_axis="y")
    _save_figure(fig, os.path.join(global_stats_folder, "global_time_of_day_importance"))


def _subject_shap_time_of_day_data(
    *,
    timeline_timestamps,
    signed_time_contribution,
    time_importance,
    invalid_mask,
):
    timeline_timestamps = np.asarray(timeline_timestamps, dtype=float)
    signed_time_contribution = np.asarray(signed_time_contribution, dtype=float)
    time_importance = np.asarray(time_importance, dtype=float)
    invalid_mask = np.asarray(invalid_mask, dtype=bool)

    valid_mask = ~invalid_mask
    if not np.any(valid_mask):
        return None

    valid_timestamps = timeline_timestamps[valid_mask]
    valid_signed = signed_time_contribution[valid_mask]
    valid_importance = time_importance[valid_mask]

    day_numbers = np.floor(valid_timestamps)
    unique_days = np.sort(np.unique(day_numbers))
    rounded_slot_values = np.round(np.mod(valid_timestamps, 1.0), 8)
    unique_slots = np.sort(np.unique(rounded_slot_values))
    slots_per_day = unique_slots.size
    predictions_per_day = valid_timestamps.size / max(unique_days.size, 1)
    if slots_per_day < 2 or predictions_per_day < 1.0:
        return None

    signed_sum = np.zeros((unique_days.size, slots_per_day), dtype=float)
    importance_sum = np.zeros((unique_days.size, slots_per_day), dtype=float)
    counts = np.zeros((unique_days.size, slots_per_day), dtype=float)

    day_idx = np.searchsorted(unique_days, day_numbers)
    slot_idx = np.searchsorted(unique_slots, rounded_slot_values)
    np.add.at(signed_sum, (day_idx, slot_idx), valid_signed)
    np.add.at(importance_sum, (day_idx, slot_idx), valid_importance)
    np.add.at(counts, (day_idx, slot_idx), 1.0)

    signed_matrix = np.divide(
        signed_sum,
        counts,
        out=np.full_like(signed_sum, np.nan),
        where=counts > 0,
    )
    importance_matrix = np.divide(
        importance_sum,
        counts,
        out=np.full_like(importance_sum, np.nan),
        where=counts > 0,
    )
    mean_signed = np.nanmean(signed_matrix, axis=0)
    mean_importance = np.nanmean(importance_matrix, axis=0)
    week_timestamps, (week_signed, week_importance) = _expand_regular_grid(
        valid_timestamps,
        valid_signed,
        valid_importance,
        axis=0,
        period=None,
    )
    week_signed = np.asarray(week_signed, dtype=float)
    week_importance = np.asarray(week_importance, dtype=float)
    week_importance_max = float(np.nanmax(week_importance)) if np.any(np.isfinite(week_importance)) else 0.0
    week_indicator = week_signed * (week_importance / (week_importance_max + DAILY_INDICATOR_EPS))
    first_day = unique_days[0]
    day_labels = [
        _day_number_label(day_value, first_day)
        for day_value in unique_days
    ]

    return {
        "day_labels": day_labels,
        "slot_values": unique_slots,
        "signed_matrix": signed_matrix,
        "mean_signed": mean_signed,
        "mean_importance": mean_importance,
        "week_timestamps": week_timestamps,
        "week_signed": week_signed,
        "week_importance": week_importance,
        "week_indicator": week_indicator,
    }


def _plot_subject_shap_time_of_day_heatmap(stats_folder, subject, subject_cycle_data):
    if subject_cycle_data is None:
        return

    signed_matrix = np.asarray(subject_cycle_data["signed_matrix"], dtype=float)
    mean_signed = np.asarray(subject_cycle_data["mean_signed"], dtype=float)
    mean_importance = np.asarray(subject_cycle_data["mean_importance"], dtype=float)
    slot_values = np.asarray(subject_cycle_data["slot_values"], dtype=float)
    day_labels = subject_cycle_data["day_labels"]
    slot_values, (signed_matrix, mean_signed, mean_importance) = _expand_regular_grid(
        slot_values,
        signed_matrix,
        mean_signed,
        mean_importance,
        axis=-1,
        period=1.0,
    )

    signed_limit = np.nanmax(np.abs(signed_matrix))
    if not np.isfinite(signed_limit) or signed_limit == 0.0:
        signed_limit = np.nanmax(np.abs(mean_signed))
    if not np.isfinite(signed_limit) or signed_limit == 0.0:
        signed_limit = 1.0

    abs_limit = np.nanmax(mean_importance)
    if not np.isfinite(abs_limit) or abs_limit == 0.0:
        abs_limit = 1.0

    fig_height = max(4.8, 2.6 + 0.35 * signed_matrix.shape[0])
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(6.8, fig_height),
        sharex=True,
        gridspec_kw={"height_ratios": [max(1.2, 0.45 * signed_matrix.shape[0]), 1.0, 1.0]},
        constrained_layout=True,
    )

    slot_hours = np.asarray(slot_values, dtype=float) * 24.0
    slot_edges = _slot_hour_edges(slot_hours)
    day_edges = np.arange(signed_matrix.shape[0] + 1, dtype=float)

    im_days = axes[0].pcolormesh(
        slot_edges,
        day_edges,
        np.ma.masked_invalid(signed_matrix),
        cmap="coolwarm",
        vmin=-signed_limit,
        vmax=signed_limit,
        shading="flat",
    )
    axes[0].set_xlim(0.0, 24.0)
    axes[0].set_yticks(np.arange(len(day_labels)) + 0.5)
    axes[0].set_yticklabels(day_labels)
    axes[0].set_title(f"Subject {subject} - SHAP impact over the daily cycle")
    axes[0].set_ylabel("Day")
    plt.colorbar(im_days, ax=axes[0], fraction=0.03, pad=0.02, label="Signed contribution")

    heatmap_y_edges = np.array([0.0, 1.0], dtype=float)
    im_mean_signed = axes[1].pcolormesh(
        slot_edges,
        heatmap_y_edges,
        np.ma.masked_invalid(mean_signed.reshape(1, -1)),
        cmap="coolwarm",
        vmin=-signed_limit,
        vmax=signed_limit,
        shading="flat",
    )
    axes[1].set_xlim(0.0, 24.0)
    axes[1].set_yticks([0.5])
    axes[1].set_yticklabels(["Mean signed"])
    plt.colorbar(im_mean_signed, ax=axes[1], fraction=0.03, pad=0.02, label="Contribution")

    im_mean_abs = axes[2].pcolormesh(
        slot_edges,
        heatmap_y_edges,
        np.ma.masked_invalid(mean_importance.reshape(1, -1)),
        cmap="magma",
        vmin=0.0,
        vmax=abs_limit,
        shading="flat",
    )
    axes[2].set_xlim(0.0, 24.0)
    axes[2].set_yticks([0.5])
    axes[2].set_yticklabels(["Mean |SHAP|"])
    axes[2].set_xlabel("Time of day")
    plt.colorbar(im_mean_abs, ax=axes[2], fraction=0.03, pad=0.02, label="Magnitude")

    tick_hours = np.arange(0, 24, 2)
    tick_labels = [f"{int(hour):02d}:00" for hour in tick_hours]
    for ax in axes:
        ax.set_xticks(tick_hours)
        ax.set_xticklabels(tick_labels)

    _save_figure(fig, os.path.join(stats_folder, f"subject_{subject}_shap_time_of_day_heatmap"))


def _plot_subject_daily_indicator(stats_folder, subject, subject_cycle_data):
    if subject_cycle_data is None:
        return

    week_timestamps = np.asarray(subject_cycle_data["week_timestamps"], dtype=float)
    week_signed = np.asarray(subject_cycle_data["week_signed"], dtype=float)
    week_importance = np.asarray(subject_cycle_data["week_importance"], dtype=float)
    week_indicator = np.asarray(subject_cycle_data["week_indicator"], dtype=float)

    signed_limit = np.nanmax(np.abs(week_signed))
    if not np.isfinite(signed_limit) or signed_limit == 0.0:
        signed_limit = 1.0

    abs_limit = np.nanmax(week_importance)
    if not np.isfinite(abs_limit) or abs_limit == 0.0:
        abs_limit = 1.0

    indicator_limit = np.nanmax(np.abs(week_indicator))
    if not np.isfinite(indicator_limit) or indicator_limit == 0.0:
        indicator_limit = 1.0

    fig = plt.figure(figsize=(7.8, 5.9), constrained_layout=True)
    grid = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 0.22, 1.1])
    ax_s = fig.add_subplot(grid[0, 0])
    ax_a = fig.add_subplot(grid[1, 0], sharex=ax_s)
    ax_link = fig.add_subplot(grid[2, 0])
    ax_i = fig.add_subplot(grid[3, 0], sharex=ax_s)
    axes = [ax_s, ax_a, ax_i]

    time_edges = _time_edges(week_timestamps)
    heatmap_y_edges = np.array([0.0, 1.0], dtype=float)

    im_s = ax_s.pcolormesh(
        time_edges,
        heatmap_y_edges,
        np.ma.masked_invalid(week_signed.reshape(1, -1)),
        cmap="coolwarm",
        vmin=-signed_limit,
        vmax=signed_limit,
        shading="flat",
    )
    ax_s.set_xlim(time_edges[0], time_edges[-1])
    ax_s.set_ylabel("Signed contribution")
    ax_s.set_yticks([])
    ax_s.set_title(f"Subject {subject} - Components of the DABxP over the week")
    _add_formula_label(ax_s, r"$S_t = \sum_f \phi_{t,f}$")
    plt.colorbar(im_s, ax=ax_s, fraction=0.03, pad=0.02)

    im_a = ax_a.pcolormesh(
        time_edges,
        heatmap_y_edges,
        np.ma.masked_invalid(week_importance.reshape(1, -1)),
        cmap="magma",
        vmin=0.0,
        vmax=abs_limit,
        shading="flat",
    )
    ax_a.set_xlim(time_edges[0], time_edges[-1])
    ax_a.set_ylabel("Attribution magnitude")
    ax_a.set_yticks([])
    _add_formula_label(ax_a, r"$A_t = \frac{1}{F} \sum_f |\phi_{t,f}|$")
    plt.colorbar(im_a, ax=ax_a, fraction=0.03, pad=0.02)

    ax_link.axis("off")
    ax_link.text(
        0.5,
        0.40,
        r"$\Downarrow$",
        ha="center",
        va="center",
        fontsize=18,
    )

    _plot_value_colored_line(
        ax_i,
        week_timestamps,
        week_indicator,
        cmap="coolwarm",
        vmin=-indicator_limit,
        vmax=indicator_limit,
        linewidth=1.6,
        outline_color="#1f1f1f",
        outline_linewidth=2.6,
    )
    ax_i.axhline(0.0, color="black", linewidth=0.8)
    ax_i.set_ylabel("DABxP")
    ax_i.set_xlabel("Time")
    ax_i.set_ylim(-1.05 * indicator_limit, 1.05 * indicator_limit)
    ax_i.set_title(f"Subject {subject} - DABxP over the week")
    _add_formula_label(ax_i, r"$\mathrm{DABxP}_t = S_t \cdot \frac{A_t}{\max_t A_t + \varepsilon}$")
    _apply_axis_style(ax_i, grid_axis="y")

    ax_s.tick_params(axis="x", labelbottom=False)
    _set_real_time_heatmap_ticks(ax_a, week_timestamps)
    _set_real_time_heatmap_ticks(ax_i, week_timestamps)

    _save_figure(fig, os.path.join(stats_folder, f"subject_{subject}_DABxP"))


def create_timestamps_list(data_folder, decimation_factor):
    patient_df = pd.read_csv(data_folder + 'week/1_week_RAW.csv', engine="pyarrow", usecols=['datetime'])
    step = max(1, int(decimation_factor))
    datetimes = pd.to_datetime(patient_df[::step]['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
    timestamps_list = matplotlib.dates.date2num(datetimes.to_numpy(dtype="datetime64[ns]"))
    return timestamps_list


def plot_dashboards(
    data_folder,
    save_folder,
    metadata,
    min_mean_test_score=None,
    window_size=None,
    decimation_factor=1,
):
    _configure_plot_style()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    timestamps_file = f"timestamps_list_decim_{int(decimation_factor)}.joblib"
    if not os.path.exists(timestamps_file):
        timestamps = create_timestamps_list(data_folder, decimation_factor)
        jl.dump(timestamps, timestamps_file)

    stats_folder = os.path.join(save_folder, "Week_stats")
    os.makedirs(stats_folder, exist_ok=True)
    global_stats_folder = os.path.join(stats_folder, "Global_explainability")
    os.makedirs(global_stats_folder, exist_ok=True)

    best_estimators_df = pd.read_csv(save_folder+'best_estimators_results.csv', index_col=0)\
                             .sort_values(by=['mean_test_score', 'std_test_score'], ascending=False)

    estimators_specs_list, estimators_list = build_estimators_list(
        best_estimators_df=best_estimators_df,
        save_folder=save_folder,
        min_mean_test_score=min_mean_test_score,
        window_size=window_size,
        decimation_factor=decimation_factor,
    )

    print('Expected estimators: ',len(estimators_specs_list))

    model_path = regressor_model_path(
        save_folder=save_folder,
        estimators_list=estimators_list,
    )
    regressor = jl.load(model_path)
    prep = regressor.named_steps["prep"]
    prep_mode = getattr(prep, "mode", "block")

    # --- SHAP setup ---
    net = regressor.named_steps["model"].module_
    net.eval()
    device = next(net.parameters()).device

    # Gradient-based explainers on GRU inputs are more reliable with the non-cuDNN autograd path.
    torch.backends.cudnn.enabled = False

    timestamps = jl.load(timestamps_file)
    subject_entries = []
    for _, subject_metadata in metadata.iterrows():
        predictions, hp_tot_list, invalid_bitmap = predict_samples(
            data_folder,
            estimators_list,
            subject_metadata,
        )
        raw_input = build_regressor_sample(
            data_folder,
            estimators_list,
            subject_metadata,
            input_type="week",
            predictions_list=predictions,
            invalid_bitmap=invalid_bitmap,
        )
        model_input = np.asarray([raw_input], dtype=object)
        regressor_sequence, regressor_sequence_scaled = _transform_regressor_input_for_shap(
            regressor,
            model_input,
        )
        subject_entries.append(
            {
                "subject_metadata": subject_metadata,
                "predictions": predictions,
                "hp_tot_list": hp_tot_list,
                "invalid_bitmap": np.asarray(invalid_bitmap, dtype=np.uint8),
                "raw_input": raw_input,
                "regressor_sequence": regressor_sequence[0],
                "regressor_sequence_scaled": regressor_sequence_scaled[0],
            }
        )

    explainer = _build_regressor_shap_explainer(
        regressor=regressor,
        background_samples=subject_entries,
        device=device,
    )

    mean_prediction_list = []
    predicted_aha_list = []
    global_subject_feature_importances = []
    global_attr_rows = []
    global_feature_rows = []
    global_subject_cycle_data = []
    global_raw_feature_names = None
    global_feature_names = None

    for entry in subject_entries:
        subject_metadata = entry["subject_metadata"]
        subject = int(subject_metadata['subject'])
        subject_stats_folder = os.path.join(stats_folder, f"subject_{subject}")
        os.makedirs(subject_stats_folder, exist_ok=True)
        predictions = entry["predictions"]
        subject_mean_predictions = entry["hp_tot_list"]
        invalid_bitmap = entry["invalid_bitmap"]
        raw_input = entry["raw_input"]
        regressor_sequence = entry["regressor_sequence"]
        regressor_sequence_scaled = entry["regressor_sequence_scaled"]
        model_input = np.asarray([raw_input], dtype=object)
        invalid_mask = np.asarray(invalid_bitmap, dtype=bool)
        window_timestamps = timestamps[::window_size][:len(predictions[0])]
        regressor_timestamps, regressor_invalid_mask, _, _ = _aggregate_window_timeline(
            prep=prep,
            window_timestamps=window_timestamps,
            invalid_mask=invalid_mask,
            window_size=window_size,
            decimation_factor=decimation_factor,
            series=None,
        )

        mag_D, mag_ND = read_file(
            data_folder,
            subject,
            window_size,
            decimation_factor,
            input_type='week',
            return_mag=1
        )

        mean_prediction_list.append(subject_mean_predictions)
        real_aha = subject_metadata['AHA']

        regressor_output = np.asarray(regressor.predict(model_input), dtype=float)
        if regressor_output.ndim == 3:
            predicted_aha = float(np.clip(regressor_output[0, -1, 0], 0, 100))
        elif regressor_output.ndim == 2:
            predicted_aha = float(np.clip(regressor_output[0, -1], 0, 100))
        else:
            predicted_aha = float(np.clip(regressor_output[0], 0, 100))
        predicted_aha_list.append(predicted_aha)

        print('Patient ', subject)
        print(' - AHA:     ', real_aha)
        print(' - Mean predictions: ', subject_mean_predictions)
        print(' - Predicted AHA from mean predictions: ', predicted_aha)

        #################### EXPLAINABILITY ####################

        x = torch.tensor(regressor_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        previous_return_last_step, previous_keep_output_dim = _enable_regressor_shap_output(net)
        try:
            shap_values = explainer.shap_values(x)
        finally:
            _restore_regressor_output_flags(net, previous_return_last_step, previous_keep_output_dim)

        attr = shap_values[0] if isinstance(shap_values, list) else shap_values
        attr = np.asarray(attr).squeeze()
        if attr.ndim == 3:
            attr = attr[..., 0]
        if attr.ndim == 1:
            if attr.size == regressor_sequence.shape[1]:
                attr = attr.reshape(1, -1)
            elif attr.size == regressor_sequence.shape[0]:
                attr = attr.reshape(-1, 1)
            else:
                attr = attr.reshape(regressor_sequence.shape)
        elif attr.ndim == 0:
            attr = attr.reshape(1, 1)

        abs_attr = np.abs(attr)
        time_importance = np.mean(abs_attr, axis=1)
        signed_time_contribution = np.sum(attr, axis=1)

        raw_feature_names = build_block_feature_names(
            n_features=regressor_sequence.shape[1],
            n_estimators=len(predictions),
        )
        feature_names = [_format_feature_name(name) for name in raw_feature_names]
        if global_raw_feature_names is None:
            global_raw_feature_names = raw_feature_names
            global_feature_names = feature_names

        if regressor_timestamps.shape[0] != signed_time_contribution.shape[0]:
            aligned_len = min(regressor_timestamps.shape[0], signed_time_contribution.shape[0])
            attr = attr[:aligned_len]
            abs_attr = abs_attr[:aligned_len]
            regressor_sequence_scaled = regressor_sequence_scaled[:aligned_len]
            regressor_timestamps = regressor_timestamps[:aligned_len]
            regressor_invalid_mask = regressor_invalid_mask[:aligned_len]
            time_importance = time_importance[:aligned_len]
            signed_time_contribution = signed_time_contribution[:aligned_len]

        valid_regressor_mask = ~regressor_invalid_mask
        if np.any(valid_regressor_mask):
            subject_feature_importance = np.nanmean(abs_attr[valid_regressor_mask], axis=0)
            global_subject_feature_importances.append(subject_feature_importance)
            global_attr_rows.append(attr[valid_regressor_mask])
            global_feature_rows.append(regressor_sequence_scaled[valid_regressor_mask])

        heatmap_timestamps, (heatmap_abs_attr,) = _expand_regular_grid(
            regressor_timestamps,
            abs_attr,
            axis=0,
            period=None,
        )
        heatmap_height = max(4.2, 1.8 + 0.22 * len(feature_names))
        fig, ax = plt.subplots(figsize=(6.6, heatmap_height), constrained_layout=True)
        vmax = np.nanpercentile(abs_attr, 99)
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        time_edges = _time_edges(heatmap_timestamps)
        feature_edges = np.arange(len(feature_names) + 1, dtype=float)
        im = ax.pcolormesh(
            time_edges,
            feature_edges,
            np.ma.masked_invalid(heatmap_abs_attr.T),
            cmap="inferno",
            vmin=0.0,
            vmax=vmax,
            shading="flat",
        )
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="|SHAP value|")
        ax.set_xlabel("Time")
        ax.set_ylabel("Feature")
        ax.set_yticks(np.arange(len(feature_names)) + 0.5)
        ax.set_yticklabels(feature_names)
        ax.set_xlim(time_edges[0], time_edges[-1])
        _set_real_time_heatmap_ticks(ax, heatmap_timestamps)
        ax.set_title(f"Subject {subject} - SHAP heatmap")
        _save_figure(fig, os.path.join(subject_stats_folder, f"subject_{subject}_explain_heatmap"))

        summary_height = max(4.2, 1.8 + 0.24 * len(feature_names))
        plt.figure(figsize=(6.4, summary_height))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The NumPy global RNG was seeded by calling `np.random.seed`.*",
                category=FutureWarning,
            )
            shap.summary_plot(
                attr,
                features=regressor_sequence_scaled,
                feature_names=feature_names,
                show=False,
                plot_size=None,
                max_display=len(feature_names),
            )
        fig = plt.gcf()
        plt.title(f"Subject {subject} - SHAP summary")
        _save_figure(fig, os.path.join(subject_stats_folder, f"subject_{subject}_shap_summary"))

        plt.figure(figsize=(6.4, summary_height))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The NumPy global RNG was seeded by calling `np.random.seed`.*",
                category=FutureWarning,
            )
            shap.summary_plot(
                attr,
                features=regressor_sequence_scaled,
                feature_names=feature_names,
                plot_type="bar",
                show=False,
                plot_size=None,
                max_display=len(feature_names),
            )
        fig = plt.gcf()
        plt.title(f"Subject {subject} - SHAP bar summary")
        _save_figure(fig, os.path.join(subject_stats_folder, f"subject_{subject}_shap_summary_bar"))

        fig, ax = _new_figure("timeline")
        ax.plot(regressor_timestamps, time_importance, color="tab:orange")
        ax.set_xlabel("Time")
        ax.set_ylabel("Mean |SHAP value|")
        ax.set_title(f"Subject {subject} - SHAP time importance")
        _format_time_axis(ax)
        _apply_axis_style(ax, grid_axis="y")
        _save_figure(fig, os.path.join(subject_stats_folder, f"subject_{subject}_explain_time"))

        bar_width = _timeline_bar_width(regressor_timestamps)
        bar_colors = [
            matplotlib.colors.to_rgba(
                "tab:red" if contribution >= 0 else "tab:blue",
                alpha=0.35 if is_invalid else 0.8,
            )
            for contribution, is_invalid in zip(signed_time_contribution, regressor_invalid_mask)
        ]

        fig, ax_contrib = _new_figure("wide")
        _apply_axis_style(ax_contrib, grid_axis="y")
        ax_contrib.bar(
            regressor_timestamps,
            signed_time_contribution,
            width=bar_width,
            color=bar_colors,
            edgecolor="none",
        )
        ax_contrib.axhline(y=0.0, color="black", linewidth=0.8)
        ax_contrib.set_ylabel("Sum SHAP")
        ax_contrib.set_xlabel("Time")
        ax_contrib.set_title(f"Subject {subject} - SHAP directional time contributions")
        _format_time_axis(ax_contrib)
        _save_figure(fig, os.path.join(subject_stats_folder, f"subject_{subject}_shap_time_direction"))

        #########################################################

        #################### WEEK MAGNITUDE ####################
        fig, ax = _new_figure("timeline")
        ax.plot(timestamps, mag_D, label="Dominant")
        ax.plot(timestamps, mag_ND, label="Non-dominant")
        ax.set_xlabel("Time")
        ax.set_ylabel("Magnitude")
        ax.set_title(f"Subject {subject} - Week magnitude")
        _format_time_axis(ax)
        _apply_axis_style(ax, grid_axis="y")
        ax.legend(loc="upper right")
        _save_figure(fig, os.path.join(subject_stats_folder, f"subject_{subject}_mag"))

        enmo_D = compute_enmo(mag_D)
        enmo_ND = compute_enmo(mag_ND)

        fig, ax = _new_figure("timeline")
        ax.plot(timestamps, enmo_D, label="Dominant")
        ax.plot(timestamps, enmo_ND, label="Non-dominant")
        ax.set_xlabel("Time")
        ax.set_ylabel("ENMO")
        ax.set_title(f"Subject {subject} - Week ENMO")
        _format_time_axis(ax)
        _apply_axis_style(ax, grid_axis="y")
        ax.legend(loc="upper right")
        _save_figure(fig, os.path.join(subject_stats_folder, f"subject_{subject}_enmo"))

        #################### WINDOW PREDICTIONS ####################
        fig, ax = _new_figure("timeline")
        ax.set_ylim([0,1])
        classifier_colors = plt.get_cmap("tab10").colors
        for i, pred in enumerate(predictions):
            ax.scatter(
                window_timestamps,
                pred,
                color=classifier_colors[i % len(classifier_colors)],
                s=6,
                alpha=0.45,
                linewidths=0,
                label=f"Classifier {i + 1}",
            )
        ax.set_xlabel("Time")
        ax.set_ylabel("Classifier output")
        ax.set_title(f"Subject {subject} - Window predictions")
        _format_time_axis(ax)
        _apply_axis_style(ax, grid_axis="y")
        if len(predictions) > 1:
            ax.legend(loc="best")
        _save_figure(fig, os.path.join(subject_stats_folder, f"subject_{subject}_samples"))

        #################### MEAN PREDICTION TIMELINE ####################

        fig, ax = _new_figure("timeline")
        ax.set_ylim([0,1])
        _format_time_axis(ax)
        _apply_axis_style(ax, grid_axis="y")
        for i, pred in enumerate(predictions):
            block_timestamps, _, _, block_mean_prediction = _aggregate_window_timeline(
                prep=prep,
                window_timestamps=window_timestamps,
                invalid_mask=invalid_mask,
                window_size=window_size,
                decimation_factor=decimation_factor,
                series=pred,
            )
            ax.plot(block_timestamps, block_mean_prediction, label=f"Classifier {i + 1}")
        if len(predictions) > 1:
            ax.legend(loc="best")
        ax.set_xlabel("Time")
        ax.set_ylabel("Mean prediction")
        ax.set_title(f"Subject {subject} - Mean prediction timeline")
        _save_figure(fig, os.path.join(subject_stats_folder, f"subject_{subject}_mean_prediction_timeline"))

        ##################### VALIDITY TIMELINE ######################
        validity_timestamps, _, block_valid_fraction, _ = _aggregate_window_timeline(
            prep=prep,
            window_timestamps=window_timestamps,
            invalid_mask=invalid_mask,
            window_size=window_size,
            decimation_factor=decimation_factor,
            series=None,
        )
        validity_timeline = 100.0 * block_valid_fraction

        fig, ax = _new_figure("timeline")
        ax.set_ylim([0,100])
        _format_time_axis(ax)
        _apply_axis_style(ax, grid_axis="y")
        ax.plot(validity_timestamps, validity_timeline, color="tab:purple")
        ax.set_xlabel("Time")
        ax.set_ylabel(_percent_label("Validity (%)"))
        ax.set_title(f"Subject {subject} - Local validity timeline")
        _save_figure(fig, os.path.join(subject_stats_folder, f"subject_{subject}_validity_timeline"))

        ##################### PREDICTED AHA PLOT ####################
        prev_return_last_step = getattr(net, "return_last_step", None)
        if prev_return_last_step is not None:
            net.return_last_step = False
        try:
            regressor_output_seq = np.asarray(regressor.predict(model_input), dtype=float)
        finally:
            if prev_return_last_step is not None:
                net.return_last_step = prev_return_last_step

        if regressor_output_seq.ndim == 3:
            aha_timeline = np.clip(regressor_output_seq[0, :, 0], 0, 100)
        elif regressor_output_seq.ndim == 2:
            aha_timeline = np.clip(regressor_output_seq[0], 0, 100)
        else:
            aha_timeline = np.clip(np.asarray(regressor_output_seq).reshape(-1), 0, 100)
        aha_timeline_valid_only = aha_timeline.copy()
        aha_timeline_valid_only[regressor_invalid_mask] = np.nan

        fig, ax = _new_figure("timeline")
        ax.set_ylim([-1,101])
        _format_time_axis(ax)
        _apply_axis_style(ax, grid_axis="y")
        ax.axhline(y=real_aha, color="tab:blue", linestyle="--", linewidth=1.0, label="AHA")
        ax.plot(regressor_timestamps, aha_timeline_valid_only, color="tab:green", label="Predicted AHA")
        ax.set_xlabel("Time")
        ax.set_ylabel("DAB")
        ax.set_title(f"Subject {subject} - DAB")
        ax.legend(loc="best")
        _save_figure(fig, os.path.join(subject_stats_folder, f"subject_{subject}_DAB"))

        subject_cycle_data = _subject_shap_time_of_day_data(
            timeline_timestamps=regressor_timestamps,
            signed_time_contribution=signed_time_contribution,
            time_importance=time_importance,
            invalid_mask=regressor_invalid_mask,
        )
        _plot_subject_shap_time_of_day_heatmap(
            subject_stats_folder,
            subject,
            subject_cycle_data,
        )
        _plot_subject_daily_indicator(
            subject_stats_folder,
            subject,
            subject_cycle_data,
        )
        if subject_cycle_data is not None:
            global_subject_cycle_data.append(subject_cycle_data)

    metadata_out = metadata.copy()
    metadata_out['mean_prediction_list'] = mean_prediction_list
    metadata_out['predicted_aha'] = predicted_aha_list
    metadata_out.to_csv(os.path.join(stats_folder, "predictions_dataframe.csv"), index=False)

    if global_feature_names is not None and global_raw_feature_names is not None:
        _plot_global_feature_importance(
            global_stats_folder,
            global_feature_names,
            global_subject_feature_importances,
        )
        _plot_global_grouped_importance(
            global_stats_folder,
            global_raw_feature_names,
            global_subject_feature_importances,
        )
        _plot_global_shap_summary(
            global_stats_folder,
            global_attr_rows,
            global_feature_rows,
            global_feature_names,
        )
    _plot_global_time_of_day_importance(
        global_stats_folder,
        global_subject_cycle_data,
    )


def plot_corrcoeff(iterations_folders:list, save_folder:str):
    _configure_plot_style()
    predictions_dataframe = pd.DataFrame()
    counter = 0
    for folder in iterations_folders:
        folder_dataframe = pd.read_csv(folder + 'Week_stats/predictions_dataframe.csv', index_col=0)
        folder_dataframe['iteration'] = counter
        predictions_dataframe = pd.concat([predictions_dataframe, folder_dataframe])
        counter += 1

    mean_prediction_column = (
        "mean_prediction_list"
        if "mean_prediction_list" in predictions_dataframe.columns
        else "healthy_percentage"
    )
    mean_prediction_lists = predictions_dataframe[mean_prediction_column].apply(json.loads).tolist()

    cdict = {0:'green', 1: 'gold', 2: 'orange', 3: 'red'}
    fig, axs = plt.subplots(1, 3, figsize=_figure_size("corrcoeff"), constrained_layout=True)

    scatter_x = np.array([])
    scatter_y = np.array([])
    scatter_marker = np.array([])
    group = np.array([])

    for sublist, aha, macs, iteration in zip(mean_prediction_lists, predictions_dataframe['AHA'].values, predictions_dataframe['MACS'].values, predictions_dataframe['iteration'].values):
        for mean_prediction in sublist:
            scatter_x = np.append(scatter_x, mean_prediction)
            scatter_y = np.append(scatter_y, aha)
            scatter_marker = np.append(scatter_marker, iteration)
            group = np.append(group, macs)

    _apply_axis_style(axs[0], grid_axis="both")
    plotted_labels = set()
    for g, m in product(np.unique(group), np.unique(scatter_marker)):
        label = 'MACS ' + str(int(g)) if g not in plotted_labels else None
        axs[0].scatter(scatter_x[(group == g) & (scatter_marker == m)],
                       scatter_y[(group == g) & (scatter_marker == m)],
                       c=cdict[g], label=label, s=50, marker="$"+str(int(m))+"$")
        plotted_labels.add(g)

    axs[0].legend()
    axs[0].set_xlabel('Mean prediction')
    axs[0].set_ylabel('AHA')

    scatter_x = np.array([sublist[0] for sublist in mean_prediction_lists])
    scatter_y = np.array(predictions_dataframe['AHA'].values)
    scatter_marker = np.array(predictions_dataframe['iteration'].values)
    group = np.array(predictions_dataframe['MACS'].values)

    _apply_axis_style(axs[1], grid_axis="both")
    plotted_labels = set()
    for g, m in product(np.unique(group), np.unique(scatter_marker)):
        label = 'MACS ' + str(g) if g not in plotted_labels else None
        axs[1].scatter(scatter_x[(group == g) & (scatter_marker == m)],
                       scatter_y[(group == g) & (scatter_marker == m)],
                       c=cdict[g], label=label, s=50, marker="$"+str(int(m))+"$")
        plotted_labels.add(g)

    axs[1].legend()
    axs[1].set_xlabel('Mean prediction')
    axs[1].set_ylabel('AHA')

    scatter_x = np.array(predictions_dataframe['predicted_aha'].values)

    _apply_axis_style(axs[2], grid_axis="both")
    plotted_labels = set()
    for g in np.unique(group):
        label = 'MACS ' + str(g) if g not in plotted_labels else None
        axs[2].scatter(scatter_x[group == g], scatter_y[group == g], c=cdict[g], label=label, s=50)
        plotted_labels.add(g)

    axs[2].legend()
    axs[2].set_xlabel('DAB')
    axs[2].set_ylabel('AHA')

    _save_figure(fig, os.path.join(save_folder, "Scatter_AHA_mean_prediction_DAB"))

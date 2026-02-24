import os
import json
import warnings
import pandas as pd
from itertools import product
import joblib as jl
import numpy as np
from predict_samples import build_estimators_list, predict_samples
import matplotlib
import matplotlib.pyplot as plt
from train_regressor import regressor_model_path, build_regressor_sequence
from read_file import read_file

import torch
import shap


def _build_feature_names(n_features, n_estimators):
    names = [f"classifier_{i}" for i in range(n_estimators)]
    names += ["invalid_flag", "time_sin", "time_cos"]
    return names[:n_features]


def create_timestamps_list(data_folder, decimation_factor):
    patient_df = pd.read_csv(data_folder + 'week/1_week_RAW.csv', engine="pyarrow", usecols=['datetime'])
    step = max(1, int(decimation_factor))
    datetimes = pd.to_datetime(patient_df[::step]['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
    timestamps_list = matplotlib.dates.date2num(datetimes.dt.to_pydatetime())
    return timestamps_list


def plot_dashboards(
    data_folder,
    save_folder,
    metadata,
    min_mean_test_score=None,
    window_size=None,
    decimation_factor=1,
):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    timestamps_file = f"timestamps_list_decim_{int(decimation_factor)}.joblib"
    if not os.path.exists(timestamps_file):
        timestamps = create_timestamps_list(data_folder, decimation_factor)
        jl.dump(timestamps, timestamps_file)

    stats_folder = save_folder + 'Week_stats/'
    os.makedirs(stats_folder, exist_ok=True)

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

    # --- SHAP setup ---
    net = regressor.module_
    torch.set_grad_enabled(True)
    net.eval()
    device = next(net.parameters()).device

    # DISABILITA CUDNN PER RNN
    torch.backends.cudnn.enabled = False

    timestamps = jl.load(timestamps_file)

    healthy_percentage = []
    predicted_aha_list = []

    for _, subject_metadata in metadata.iterrows():
        subject = int(subject_metadata['subject'])

        predictions, hp_tot_list, invalid_bitmap = predict_samples(
            data_folder,
            estimators_list,
            subject_metadata,
        )

        invalid_mask = np.asarray(invalid_bitmap, dtype=bool)
        regressor_sequence = build_regressor_sequence(predictions, invalid_bitmap, window_size, decimation_factor)
        window_timestamps = timestamps[::window_size][:regressor_sequence.shape[0]]

        mag_D, mag_ND = read_file(
            data_folder,
            subject,
            window_size,
            decimation_factor,
            input_type='week',
            return_mag=1
        )

        healthy_percentage.append(hp_tot_list)
        real_aha = subject_metadata['AHA']

        predicted_aha = np.asarray(regressor.predict(regressor_sequence[np.newaxis, :, :]), dtype=float).squeeze()
        predicted_aha = float(np.clip(predicted_aha, 0, 100))
        predicted_aha_list.append(predicted_aha)

        print('Patient ', subject)
        print(' - AHA:     ', real_aha)
        print(' - HP:      ', hp_tot_list)
        print(' - AHA predicted from HP: ', predicted_aha)

        #################### EXPLAINABILITY ####################

        x = torch.tensor(regressor_sequence, dtype=torch.float32).unsqueeze(0).to(device)
        baseline = x.mean(dim=1, keepdim=True).repeat(1, x.shape[1], 1)
        explainer = shap.GradientExplainer(net, baseline)
        shap_values = explainer.shap_values(x)
        attr = shap_values[0] if isinstance(shap_values, list) else shap_values
        attr = np.asarray(attr).squeeze()
        if attr.ndim == 3:
            attr = attr[..., 0]

        abs_attr = np.abs(attr)
        time_importance = np.mean(abs_attr, axis=1)

        feature_names = _build_feature_names(
            n_features=regressor_sequence.shape[1],
            n_estimators=len(predictions),
        )

        plt.figure(figsize=(10,4))
        plt.imshow(abs_attr.T, aspect="auto", cmap="inferno", vmin=0.0, vmax=np.percentile(abs_attr, 99))
        plt.colorbar(label="|SHAP value|")
        plt.xlabel("Time window")
        plt.ylabel("Feature")
        plt.title(f"Subject {subject} - SHAP heatmap")
        plt.tight_layout()
        plt.savefig(stats_folder + f"subject_{subject}_explain_heatmap.png", dpi=300)
        plt.close()

        plt.figure(figsize=(8, 5))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The NumPy global RNG was seeded by calling `np.random.seed`.*",
                category=FutureWarning,
            )
            shap.summary_plot(
                attr,
                features=regressor_sequence,
                feature_names=feature_names,
                show=False,
                plot_size=None,
            )
        plt.title(f"Subject {subject} - SHAP summary")
        plt.tight_layout()
        plt.savefig(stats_folder + f"subject_{subject}_shap_summary.png", dpi=300)
        plt.close()

        plt.figure(figsize=(8, 5))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The NumPy global RNG was seeded by calling `np.random.seed`.*",
                category=FutureWarning,
            )
            shap.summary_plot(
                attr,
                features=regressor_sequence,
                feature_names=feature_names,
                plot_type="bar",
                show=False,
                plot_size=None,
            )
        plt.title(f"Subject {subject} - SHAP bar summary")
        plt.tight_layout()
        plt.savefig(stats_folder + f"subject_{subject}_shap_summary_bar.png", dpi=300)
        plt.close()

        plt.figure(figsize=(8,3))
        plt.plot(time_importance)
        plt.xlabel("Time window")
        plt.ylabel("Mean |SHAP value|")
        plt.title(f"Subject {subject} - SHAP time importance")
        plt.tight_layout()
        plt.savefig(stats_folder + f"subject_{subject}_explain_time.png", dpi=300)
        plt.close()

        #########################################################

        #################### ANDAMENTO WEEK GENERALE ####################
        plt.grid()
        ax = plt.gca()
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        plt.plot(timestamps, mag_D)
        plt.plot(timestamps, mag_ND)
        plt.xlabel("Orario")
        plt.ylabel("Magnitudo")
        plt.gcf().set_size_inches(8, 2)
        plt.tight_layout()
        plt.savefig(stats_folder + 'subject_' +str(subject)+'_mag.png', dpi = 500)
        plt.close()

        #################### GRAFICO DEI PUNTI ####################
        ax = plt.gca()
        ax.set_ylim([0,1])
        for pred in predictions:
            plt.scatter(list(range(len(pred))), pred, c=pred, cmap='viridis', norm=plt.Normalize(0, 1), s=1)

        plt.xlabel("Sample")
        plt.ylabel("Classificazione")
        plt.gcf().set_size_inches(8, 2)
        plt.tight_layout()
        plt.savefig(stats_folder + '/subject_' +str(subject)+'_samples.png', dpi = 500)
        plt.close()

        #################### CPI TIMELINE ####################
        valid_per_window = (~invalid_mask).astype(float)
        cumulative_valid_count = np.cumsum(valid_per_window)

        plt.grid()
        ax = plt.gca()
        ax.set_ylim([0,1])
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        for i, pred in enumerate(predictions):
            pred = np.asarray(pred, dtype=float)
            cumulative_valid_sum = np.cumsum(np.where(~invalid_mask, pred, 0.0))
            cpi_timeline = np.divide(
                cumulative_valid_sum,
                cumulative_valid_count,
                out=np.full(cumulative_valid_sum.shape, np.nan, dtype=float),
                where=cumulative_valid_count > 0,
            )
            plt.plot(window_timestamps, cpi_timeline, label=f"classifier_{i}")
        if len(predictions) > 1:
            plt.legend()
        plt.xlabel("Orario")
        plt.ylabel("CPI")
        plt.gcf().set_size_inches(8, 2)
        plt.tight_layout()
        plt.savefig(stats_folder + '/subject_' +str(subject)+'_CPI_timeline.png', dpi = 500)
        plt.close()

        ##################### SIGNIFICATIVITY TIMELINE ######################
        significance_timeline = (100.0 * cumulative_valid_count) / np.arange(1, len(invalid_mask) + 1, dtype=float)

        plt.grid()
        ax = plt.gca()
        ax.set_ylim([0,100])
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        plt.plot(window_timestamps, significance_timeline)
        plt.xlabel("Orario")
        plt.ylabel("Significatività")
        plt.gcf().set_size_inches(8, 2)
        plt.tight_layout()
        plt.savefig(stats_folder + '/subject_' +str(subject)+'_validity_timeline.png', dpi = 500)
        plt.close()

        ##################### PREDICTED AHA PLOT ####################
        aha_prefix = []
        for stop in range(1, regressor_sequence.shape[0] + 1):
            seq_prefix = regressor_sequence[:stop]
            predicted_prefix_aha = np.asarray(regressor.predict(seq_prefix[np.newaxis, :, :]), dtype=float).squeeze()
            aha_prefix.append(float(np.clip(predicted_prefix_aha, 0, 100)))

        plt.grid()
        ax = plt.gca()
        ax.set_ylim([-1,101])
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        plt.axhline(y = real_aha, color = 'b', linestyle = '--', linewidth= 1, label='AHA')
        plt.xlabel("Orario")
        plt.ylabel("Home-AHA")
        plt.plot(window_timestamps, aha_prefix, c='green')
        plt.legend()
        plt.gcf().set_size_inches(8, 2)
        plt.tight_layout()
        plt.savefig(stats_folder + '/subject_' +str(subject)+'_Home-AHA.png', dpi = 500)
        plt.close()

    metadata_out = metadata.copy()
    metadata_out['healthy_percentage'] = healthy_percentage
    metadata_out['predicted_aha'] = predicted_aha_list
    metadata_out.to_csv(stats_folder + '/predictions_dataframe.csv', index=False)


def plot_corrcoeff(iterations_folders:list, save_folder:str):
    predictions_dataframe = pd.DataFrame()
    counter = 0
    for folder in iterations_folders:
        folder_dataframe = pd.read_csv(folder + 'Week_stats/predictions_dataframe.csv', index_col=0)
        folder_dataframe['iteration'] = counter
        predictions_dataframe = pd.concat([predictions_dataframe, folder_dataframe])
        counter += 1

    CPI_list_list = predictions_dataframe['healthy_percentage'].apply(json.loads).tolist()

    cdict = {0:'green', 1: 'gold', 2: 'orange', 3: 'red'}
    _, axs = plt.subplots(1, 3, figsize=(15, 5)) 

    scatter_x = np.array([])
    scatter_y = np.array([])
    scatter_marker = np.array([])
    group = np.array([])

    for sublist, aha, macs, iteration in zip(CPI_list_list, predictions_dataframe['AHA'].values, predictions_dataframe['MACS'].values, predictions_dataframe['iteration'].values):
        for cpi in sublist:
            scatter_x = np.append(scatter_x, cpi)
            scatter_y = np.append(scatter_y, aha)
            scatter_marker = np.append(scatter_marker, iteration)
            group = np.append(group, macs)

    axs[0].grid()
    plotted_labels = set()
    for g, m in product(np.unique(group), np.unique(scatter_marker)):
        label = 'MACS ' + str(int(g)) if g not in plotted_labels else None
        axs[0].scatter(scatter_x[(group == g) & (scatter_marker == m)],
                       scatter_y[(group == g) & (scatter_marker == m)],
                       c=cdict[g], label=label, s=50, marker="$"+str(int(m))+"$")
        plotted_labels.add(g)

    axs[0].legend()
    axs[0].set_xlabel('CPI')
    axs[0].set_ylabel('AHA')

    scatter_x = np.array([sublist[0] for sublist in CPI_list_list])
    scatter_y = np.array(predictions_dataframe['AHA'].values)
    scatter_marker = np.array(predictions_dataframe['iteration'].values)
    group = np.array(predictions_dataframe['MACS'].values)

    axs[1].grid()
    plotted_labels = set()
    for g, m in product(np.unique(group), np.unique(scatter_marker)):
        label = 'MACS ' + str(g) if g not in plotted_labels else None
        axs[1].scatter(scatter_x[(group == g) & (scatter_marker == m)],
                       scatter_y[(group == g) & (scatter_marker == m)],
                       c=cdict[g], label=label, s=50, marker="$"+str(int(m))+"$")
        plotted_labels.add(g)

    axs[1].legend()
    axs[1].set_xlabel('CPI')
    axs[1].set_ylabel('AHA')

    scatter_x = np.array(predictions_dataframe['predicted_aha'].values)

    axs[2].grid()
    plotted_labels = set()
    for g in np.unique(group):
        label = 'MACS ' + str(g) if g not in plotted_labels else None
        axs[2].scatter(scatter_x[group == g], scatter_y[group == g], c=cdict[g], label=label, s=50)
        plotted_labels.add(g)

    axs[2].legend()
    axs[2].set_xlabel('Home-AHA')
    axs[2].set_ylabel('AHA')

    plt.savefig(save_folder+'Scatter_AHA_CPI_Home-AHA.png', dpi=500)
    plt.close()

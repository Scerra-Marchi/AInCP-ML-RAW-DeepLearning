import os
import json
import pandas as pd
from itertools import product
from sktime.base import BaseEstimator
import joblib as jl
import numpy as np
from predict_samples import predict_samples
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import hashlib
import math
#import warnings 


def create_timestamps_list(data_folder):
    patient_df = pd.read_csv(data_folder + 'week/1_week_RAW.csv')  # I pazienti hanno tutti lo stesso numero di campioni
    datetimes = pd.to_datetime(patient_df[::3]['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
    timestamps_list = matplotlib.dates.date2num(datetimes.dt.to_pydatetime())
    return timestamps_list


def plot_dashboards(data_folder, save_folder, subjects_indexes, min_mean_test_score, window_size, decimation_factor):
        
    #warnings.filterwarnings("ignore")

    # Cambio la directory di esecuzione in quella dove si trova questo file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Creazione lista timestamps
    timestamps_file = 'timestamps_list'
    if not os.path.exists(timestamps_file):
        timestamps = create_timestamps_list(data_folder)
        # print('Lunghezza lista timestamps:', len(timestamps))
        jl.dump(timestamps, timestamps_file)    # La lista dei timestamp viene salvata in un file all'interno della cartella corrente

    stats_folder = save_folder + 'Week_stats/'

    best_estimators_df = pd.read_csv(save_folder+'best_estimators_results.csv', index_col=0).sort_values(by=['mean_test_score', 'std_test_score'], ascending=False)

    #estimators_specs_list = []
    #estimators_specs_list.append(best_estimators_df[best_estimators_df['method'] == 'concat'].iloc[0])
    #estimators_specs_list.append(best_estimators_df[best_estimators_df['method'] == 'ai'].iloc[0])
    #estimators_specs_list.append(best_estimators_df[best_estimators_df['method'] == 'difference'].iloc[0])
    #estimators_specs_list = [row for index, row in best_estimators_df[(best_estimators_df['mean_test_score'] >= 0.975) & (best_estimators_df['window_size'] == 600)].iterrows()]
    #estimators_specs_list = [row for index, row in best_estimators_df[(best_estimators_df['mean_test_score'] == 1) & (best_estimators_df['method'] == 'difference')].iterrows()]

    #estimators_specs_list = [row for index, row in best_estimators_df[(best_estimators_df['mean_test_score'] >= 0.954) & (best_estimators_df['window_size'] == 300)].iterrows()]
    estimators_specs_list = [row for index, row in best_estimators_df[(best_estimators_df['mean_test_score'] >= min_mean_test_score) & (best_estimators_df['window_size'] == window_size) & (best_estimators_df['decimation_factor'] == decimation_factor)].iterrows()]
    
    print('Expected estimators: ',len(estimators_specs_list))
    estimators_list = []
    model_id_concat = ''
    
    for estimators_specs in estimators_specs_list:
        estimator_dir = save_folder + "Trained_models/" + estimators_specs['method'] + "/" + str(estimators_specs['window_size']) + "_points/" + str(estimators_specs['decimation_factor']) + "_decimation_factor/" + estimators_specs['model_type'].split(".")[-1] + "/gridsearch_" + estimators_specs['gridsearch_hash']  + "/"

        with open(estimator_dir + 'GridSearchCV_stats/best_estimator_stats.json', "r") as stats_f:
            grid_search_best_params = json.load(stats_f)
        
        print('Loading -> ', estimator_dir + 'best_estimator.zip')
        estimator = BaseEstimator().load_from_path(estimator_dir + 'best_estimator.zip')
        estimators_list.append({'estimator': estimator, 'method': estimators_specs['method'], 'window_size': estimators_specs['window_size'], 'decimation_factor': estimators_specs['decimation_factor'], 'hemi_cluster': grid_search_best_params['Hemi cluster']})
        print('Loaded -> ', estimator_dir + 'best_estimator.zip')
        model_id_concat = model_id_concat + str(estimator.get_params())

    metadata = pd.read_excel(data_folder + 'metadata2023_08.xlsx').iloc[subjects_indexes]
    metadata.drop(['dom', 'date AHA', 'start AHA', 'stop AHA'], axis=1, inplace=True) # 'age_aha', 'gender', 

    reg_path = save_folder + 'Regressors/regressor_'+ (hashlib.sha256((model_id_concat).encode()).hexdigest()[:10])
    regressor = jl.load(reg_path)

    os.makedirs(stats_folder, exist_ok=True)
    timestamps = jl.load(timestamps_file)     # Si carica la lista dei timestamps

    ds_freq = 80/decimation_factor   # Frequenza di campionamento del segnale decimato (Hz)
    sample_size = math.ceil(window_size / ds_freq)   # Dimensione IN SECONDI del campione (finestra) -> 6400 / 26.67 ≃ 240 secondi

    trend_block_size = int((60 * 60 * 6) / sample_size)  # Numero di finestre raggruppate in un blocco da 6 ore
    block_samples = int(6 * 60 * 60 * ds_freq)      # Numero di campionamenti in 6 ore
    significativity_threshold = 75                  # Percentuale di finestre in un blocco che devono essere prese per renderlo significativo

    plot_show = False
    '''
    answer = input("Do you want to see the dashboard for each patient? (yes/no): \n")
    # If the user enters "yes", show the plot
    if answer.lower() == "yes":
        plot_show = True
    '''

    healthy_percentage = []
    predicted_aha_list = []

    for index in range(len(metadata)):
        subject = metadata['subject'].iloc[index]
        predictions, hp_tot_list, magnitude_D, magnitude_ND = predict_samples(data_folder, estimators_list, subject)
        healthy_percentage.append(hp_tot_list)
        real_aha = metadata['AHA'].iloc[index]
        predicted_aha = regressor.predict(np.array([hp_tot_list]))[0]
        predicted_aha = 100 if predicted_aha > 100 else predicted_aha
        predicted_aha_list.append(predicted_aha)

        print('Patient ', subject)
        print(' - AHA:     ', real_aha)
        print(' - HP:      ', hp_tot_list)
        print(' - AHA predicted from HP: ', predicted_aha)

        #################### ANDAMENTO WEEK GENERALE ####################

        #plt.title('Andamento magnitudo')
        plt.grid()
        ax = plt.gca()
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        plt.plot(timestamps, magnitude_D)
        plt.plot(timestamps, magnitude_ND)
        plt.xlabel("Orario")
        plt.ylabel("Magnitudo")
        plt.gcf().set_size_inches(8, 2)
        plt.tight_layout()
        plt.savefig(stats_folder + 'subject_' +str(subject)+'_mag.png', dpi = 500)
        plt.close()

        # Fase di plotting
        #fig, axs = plt.subplots(7)
        #fig.suptitle('Patient ' + str(i) + ' week trend, AHA: ' + str(real_aha))
            #axs[0].xaxis.set_minor_locator(matplotlib.dates.HourLocator())
            #axs[0].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(12))
            #axs[0].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(n=6))
            #axs[0].xaxis.set_minor_formatter(matplotlib.dates.DateFormatter('%H-%M'))
        #axs[0].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d-%H:%M'))
        #axs[0].plot(timestamps, magnitude_D)
        #axs[0].plot(timestamps, magnitude_ND)


        ########################## AI PLOT ##########################
        ai_list = []
        subList_magD = [magnitude_D[n:n+block_samples] for n in range(0, len(magnitude_D), block_samples)]
        subList_magND = [magnitude_ND[n:n+block_samples] for n in range(0, len(magnitude_ND), block_samples)]
        for l in range(len(subList_magD)):
            if (subList_magD[l].mean() + subList_magND[l].mean()) == 0:
                ai_list.append(np.nan)
            else:
                ai_list.append(((subList_magD[l].mean() - subList_magND[l].mean()) / (subList_magD[l].mean() + subList_magND[l].mean())) * 100)

        #axs[1].grid()
        #axs[1].set_ylim([-101,101])
        #axs[1].plot(ai_list)

        #plt.title('Andamento AI')
        plt.xlabel("Orario")
        plt.ylabel("Asimmetry Index")
        plt.grid()
        ax = plt.gca()
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        plt.plot(timestamps[::block_samples], ai_list)
        plt.gcf().set_size_inches(8, 2)
        plt.tight_layout()
        plt.savefig(stats_folder + '/subject_' +str(subject)+'_AI.png', dpi = 500)
        plt.close()


        #################### GRAFICO DEI PUNTI ####################
        for pred in predictions:
            #axs[2].scatter(list(range(len(pred))), pred, c=pred, cmap='brg', s=1) 
            plt.scatter(list(range(len(pred))), pred, c=pred, cmap='brg', norm=plt.Normalize(-1, +1), s=1)

        #plt.title('Grafico delle predizioni')
        plt.xlabel("Sample")
        plt.ylabel("Classificazione")
        plt.gcf().set_size_inches(8, 2)
        plt.tight_layout()
        plt.savefig(stats_folder + '/subject_' +str(subject)+'_samples.png', dpi = 500)
        plt.close()

        #################### ANDAMENTO A BLOCCHI ####################

        for pred in predictions:
            h_perc_list = []
            subList = [pred[n:n+trend_block_size] for n in range(0, len(pred), trend_block_size)]
            for l in subList:
                n_hemi = l.tolist().count(-1)
                n_healthy = l.tolist().count(1)
                if (((n_hemi + n_healthy) / trend_block_size) * 100) < significativity_threshold:
                    h_perc_list.append(np.nan)
                else:
                    h_perc_list.append((n_healthy / (n_hemi + n_healthy)) * 100)

            #h_perc_list.append(h_perc_list[-1]) PER LA LINEA ORIZZONTALE FINALE
            #axs[4].grid()
            #axs[4].set_ylim([-1,101])
            #axs[4].plot(h_perc_list, drawstyle = 'steps-post')
            plt.grid()
            ax = plt.gca()
            ax.set_ylim([-1,101])
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
            plt.plot(timestamps[::block_samples], h_perc_list, drawstyle = 'steps-post')
            
        #plt.title('Andamento CPI su finestre disgiunte')
        plt.xlabel("Orario")
        plt.ylabel("CPI")
        plt.gcf().set_size_inches(8, 2)
        plt.tight_layout()
        plt.savefig(stats_folder + '/subject_' +str(subject)+'_CPIblocks.png', dpi = 500)
        plt.close()
        
        ##################### ANDAMENTO SMOOTH ######################
        h_perc_list_smooth_list = []
        #plt.title('Andamento CPI su finestra scorrevole')
        plt.grid()
        ax = plt.gca()
        ax.set_ylim([-1,101])
        for pred in predictions:
            h_perc_list_smooth = []
            h_perc_list_smooth_significativity = []
            subList_smooth = [pred[n:n+trend_block_size] for n in range(0, len(pred)-trend_block_size+1)]
            for l in subList_smooth:
                n_hemi = l.tolist().count(-1)
                n_healthy = l.tolist().count(1)
                h_perc_list_smooth_significativity.append(((n_hemi + n_healthy) / trend_block_size) * 100)
                if (((n_hemi + n_healthy) / trend_block_size) * 100) < significativity_threshold:
                    h_perc_list_smooth.append(np.nan)
                else:
                    h_perc_list_smooth.append((n_healthy / (n_hemi + n_healthy)) * 100)

            #axs[5].plot(h_perc_list_smooth)

            h_perc_list_smooth_list.append(h_perc_list_smooth)

            plot_h_perc_list_smooth = [np.nan] * (trend_block_size - 1) + h_perc_list_smooth

            ax = plt.gca()
            ax.set_ylim([-1,101])
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
            plt.plot(timestamps[::window_size], plot_h_perc_list_smooth)


        plt.xlabel("Orario")
        plt.ylabel("CPI")
        plt.gcf().set_size_inches(8, 2)
        plt.tight_layout()
        plt.savefig(stats_folder + '/subject_' +str(subject)+'_CPIsmooth.png', dpi = 500)
        plt.close()

        ##################### SIGNIFICATIVITY PLOT ####################

        #plt.title('Grafico della significatività')
        plt.grid()
        ax = plt.gca()
        ax.set_ylim([-1,101])
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        plt.axhline(y = significativity_threshold, color = 'r', linestyle = '-', label='Soglia')
        plot_h_perc_list_smooth_significativity = [np.nan] * (trend_block_size - 1) + h_perc_list_smooth_significativity
        plt.plot(timestamps[::window_size], plot_h_perc_list_smooth_significativity)
        plt.legend()
        plt.xlabel("Orario")
        plt.ylabel("Significatività")
        plt.gcf().set_size_inches(8, 2)
        plt.tight_layout()
        plt.savefig(stats_folder + '/subject_' +str(subject)+'_sig.png', dpi = 500)
        plt.close()

        ##################### PREDICTED AHA PLOT ####################


        aha_list_smooth = []
        for elements in zip(*h_perc_list_smooth_list):
            if np.isnan(elements[0]):
                aha_list_smooth.append(np.nan)
            else:
                predicted_window_aha = regressor.predict(np.array([elements]))[0]
                aha_list_smooth.append(predicted_window_aha if predicted_window_aha <= 100 else 100)

        #plt.title('Andamento Home-AHA')
        conf = 5
        plt.grid()
        ax = plt.gca()
        ax.set_ylim([-1,101])
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        plt.axhline(y = real_aha, color = 'b', linestyle = '--', linewidth= 1, label='AHA')
        plt.xlabel("Orario")
        plt.ylabel("Home-AHA")
        plot_aha_list_smooth = [np.nan] * (trend_block_size - 1) + aha_list_smooth
        # switch to green
        plt.plot(timestamps[::window_size],plot_aha_list_smooth, c = 'green') 
        # will comment this stuff
        #plt.plot(timestamps[::window_size],[x if real_aha + conf < x else np.nan for x in plot_aha_list_smooth], c ='green')
        #plt.plot(timestamps[::window_size],[x if real_aha + 2*conf < x else np.nan for x in plot_aha_list_smooth], c ='darkgreen')
        #plt.plot(timestamps[::window_size],[x if x < real_aha - conf else np.nan for x in plot_aha_list_smooth], c ='orange')
        #plt.plot(timestamps[::window_size],[x if x < real_aha - 2*conf else np.nan for x in plot_aha_list_smooth], c ='darkorange')
        # until here
        plt.legend()
        plt.gcf().set_size_inches(8, 2)
        plt.tight_layout()
        plt.savefig(stats_folder + '/subject_' +str(subject)+'_Home-AHA.png', dpi = 500)
        plt.close()
        
        #############################################################
        
        #plt.savefig(stats_folder + '/subject_' +str(i)+'.png', dpi = 500)

        if(plot_show == True):
            plt.show() 
        plt.close()

    metadata['healthy_percentage'] = healthy_percentage
    metadata['predicted_aha'] = predicted_aha_list

    metadata.to_csv(stats_folder + '/predictions_dataframe.csv')

    #metadata.plot.scatter(x='healthy_percentage', y='AHA', c='MACS', colormap='viridis').get_figure().savefig(stats_folder + 'plot_healthyPerc_AHA.png')
    #metadata.plot.scatter(x='healthy_percentage', y='AI_week', c='MACS', colormap='viridis').get_figure().savefig(stats_folder + 'plot_healthyPerc_AI_week.png')
    #metadata.plot.scatter(x='healthy_percentage', y='AI_aha', c='MACS', colormap='viridis').get_figure().savefig(stats_folder + 'plot_healthyPerc_AI_aha.png')


    #print("Coefficiente di Pearson tra hp e aha:          ", (np.corrcoef(metadata['healthy_percentage'], metadata['AHA'].values))[0][1])


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

    ############################# multi CPI-AHA ################################

    # Grafico a dispersione di tutti i punti CPI-AHA per tutti i pazienti e tutte le iterazioni
    scatter_x = np.array([])
    scatter_y = np.array([])
    scatter_marker = np.array([])
    group = np.array([])    # Livelli MACS

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
        axs[0].scatter(scatter_x[(group == g) & (scatter_marker == m)], scatter_y[(group == g) & (scatter_marker == m)], c=cdict[g], label=label, s=50, marker="$"+str(int(m))+"$")
        plotted_labels.add(g)

    axs[0].legend()
    axs[0].set_xlabel('CPI')
    axs[0].set_ylabel('AHA')

    ############################# CPI-AHA ################################

    # Grafico a dispersione del primo CPI di ogni paziente per ogni iterazione contro rispettivo AHA
    scatter_x = np.array([sublist[0] for sublist in CPI_list_list])
    scatter_y = np.array(predictions_dataframe['AHA'].values)
    scatter_marker = np.array(predictions_dataframe['iteration'].values)
    group = np.array(predictions_dataframe['MACS'].values)

    axs[1].grid()
    plotted_labels = set()
    for g, m in product(np.unique(group), np.unique(scatter_marker)):
        label = 'MACS ' + str(g) if g not in plotted_labels else None
        axs[1].scatter(scatter_x[(group == g) & (scatter_marker == m)], scatter_y[(group == g) & (scatter_marker == m)], c=cdict[g], label=label, s=50, marker="$"+str(int(m))+"$")
        plotted_labels.add(g)

    axs[1].legend()
    axs[1].set_xlabel('CPI')
    axs[1].set_ylabel('AHA')

    ############################# HOME-AHA ################################

    #Grafico a dispersione di Home-AHA vs AHA
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
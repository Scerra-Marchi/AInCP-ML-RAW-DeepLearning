import os
import argparse
import numpy as np
import pandas as pd
from train_select_classifiers import train_select_classifiers
from train_regressor import train_regressor
from test_classifier_regressor import test_classifier_regressor
from plotting import plot_dashboards, plot_corrcoeff
from sklearn.model_selection import StratifiedShuffleSplit

# Cambio la directory di esecuzione in quella dove si trova questo file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#data_folder = 'C:/Users/david/Documents/University/Borsa di Studio - REDCap/only_AC-80_patients/'
data_folder = '../Dataset/'

window_size = 6400  # 4800 ≃ 180s, 6400 ≃ 240s, 8000 ≃ 300s
decimation_factor = 3
l_window_size = [window_size]  # 4800 ≃ 180s, 6400 ≃ 240s, 8000 ≃ 300s
l_decimation_factor = [decimation_factor]

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

save_folder = 'Best_model_debug/' if args.debug else 'Best_model/'
min_mean_test_score = 0.6 if args.debug else 0.85  # TODO: change to 0.85

metadata = pd.read_excel(data_folder + 'metadata2023_08.xlsx')
subjects_indexes = list(range(len(metadata)))

# Random state = 42
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train_indexes, test_indexes = next(sss.split(metadata, metadata['hemi']))

if args.debug:
    train_indexes = [0, 1, 2, 3, 4]
    test_indexes = [0, 1]


if not os.path.exists(save_folder):

    print(' ----- TRAINING CLASSIFIERS ----- ')

    train_select_classifiers(
        data_folder,
        save_folder=save_folder,
        subjects_indexes=train_indexes,
        l_window_size=l_window_size,
        l_method=['concat', 'difference', 'ai', 'enmo', 'raw'],
        l_decimation_factor=l_decimation_factor
    )
    
if not os.path.exists(save_folder + 'Regressors/'):

    print(' ----- TRAINING REGRESSOR ----- ')
    
    train_regressor(data_folder, save_folder=save_folder, train_indexes=train_indexes, min_mean_test_score=min_mean_test_score, window_size=window_size, decimation_factor=decimation_factor)

if not os.path.exists(save_folder + 'combined_test_stats.json'):

    print(' ----- TESTING CLASSIFIER AND REGRESSOR ----- ')
    
    test_classifier_regressor(data_folder, save_folder=save_folder, test_indexes=test_indexes, min_mean_test_score=min_mean_test_score, window_size=window_size, decimation_factor=decimation_factor)

if not os.path.exists(save_folder + 'Week_stats/'):
    
    print(' ----- CREATING DASHBOARDS ----- ')
    
    plot_dashboards(data_folder, save_folder=save_folder, subjects_indexes=test_indexes, min_mean_test_score=min_mean_test_score, window_size=window_size, decimation_factor=decimation_factor)
    plot_corrcoeff(iterations_folders=[save_folder], save_folder=save_folder)
import os
import json
import multiprocessing
import numpy as np
import pandas as pd
import joblib as jl
from sklearn.model_selection import RepeatedStratifiedKFold
from train_select_classifiers import train_select_classifiers
from train_regressor import train_regressor
from test_classifier_regressor import test_classifier_regressor
from plotting import plot_dashboards, plot_corrcoeff

# Cambio la directory di esecuzione in quella dove si trova questo file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#data_folder = 'C:/Users/david/Documents/University/Borsa di Studio - REDCap/only_AC-80_patients/'
data_folder = '../../../Dati_RAW/'

number_of_iterations = 5
min_mean_test_score = 0.7
window_size = 6400

folder = 'Iterations/'

# New
if not os.path.exists(folder):

    print(' ----- CREATING ITERATIONS FOLDERS AND TRAINING CLASSIFIERS ----- ')
    processes = []
    iteration = 0
    metadata = pd.read_excel(data_folder + 'metadata2023_08.xlsx')
    labels = metadata['hemi'].values    # Maybe it is better to straify on AHA as well?
    rskf = RepeatedStratifiedKFold(n_splits=number_of_iterations, n_repeats=1, random_state=42)

    for train_indexes, test_indexes in rskf.split(np.empty(metadata.shape[0]), labels):

        # Creating a structured dictionary
        data = {
            "Iteration": iteration,
            "Train Indexes": train_indexes.tolist(),
            "Test Indexes": test_indexes.tolist()
        }

        save_folder = folder + 'Iteration_' + str(iteration) + '/'
        os.makedirs(save_folder)
        iteration += 1
        
        # Writing to a JSON file
        with open(save_folder + 'iteration_data.json', 'w') as file:
            json.dump(data, file, indent=4)
        
        p = multiprocessing.Process(target=train_select_classifiers, args=(data_folder, save_folder, train_indexes, [window_size]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if not os.path.exists(folder + 'Iteration_0/Regressors/'):

    print(' ----- TRAINING REGRESSORS ----- ')
    processes = []

    for iteration in range(number_of_iterations):

        save_folder = folder + 'Iteration_' + str(iteration) + '/'

        # Reading from a JSON file and accessing data
        with open(save_folder + 'iteration_data.json', 'r') as file:
            data = json.load(file)
        retrieved_train_indexes = data['Train Indexes']

        p = multiprocessing.Process(target=train_regressor, args=(data_folder, save_folder, retrieved_train_indexes, min_mean_test_score, window_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if not os.path.exists(folder + 'Iteration_0/combined_test_stats.json'):

    print(' ----- TESTING CLASSIFIER AND REGRESSOR ----- ')
    processes = []

    for iteration in range(number_of_iterations):

        save_folder = folder + 'Iteration_' + str(iteration) + '/'

        # Reading from a JSON file and accessing data
        with open(save_folder + 'iteration_data.json', 'r') as file:
            data = json.load(file)
        retrieved_test_indexes = data['Test Indexes']

        p = multiprocessing.Process(target=test_classifier_regressor, args=(data_folder, save_folder, retrieved_test_indexes, min_mean_test_score, window_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

folder_prefix= folder + "Iteration_"
r2_list = []
corrcoef_list = []

for i in range(number_of_iterations):
    folder_name = f"{folder_prefix}{i}"
    json_file_path = os.path.join(folder_name, "combined_test_stats.json")

    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
            r2_list.append(data['Regressor Stats']['R2 Score'])
            corrcoef_list.append(data['Best Classifier Stats']['Correlation Coefficient'])

average_r2_score = np.mean(r2_list)
average_corr_score = np.mean(corrcoef_list)

print(f"The average r2 score for the regressor is: {average_r2_score}")
print(f"The average correlation CPI-AHA is: {average_corr_score}")

results = {
    "R2 Score List":r2_list,
    "Correlation List":corrcoef_list,
    "Average R2 Score": average_r2_score,
    "Average CPI-AHA Correlation": average_corr_score
}

# Writing to a JSON file
with open(folder + 'test_results.json', 'w') as file:
    json.dump(results, file, indent=4)

if not os.path.exists(folder + 'Iteration_0/Week_stats/predictions_dataframe.csv'):

    print(' ----- PLOTTING PREDICTIONS ----- ')
    processes = []


    for iteration in range(number_of_iterations):

        save_folder = folder + 'Iteration_' + str(iteration) + '/'

        with open(save_folder + 'iteration_data.json', 'r') as file:
            data = json.load(file)
        retrieved_test_indexes = data['Test Indexes']

        p = multiprocessing.Process(target=plot_dashboards, args=(data_folder, save_folder, retrieved_test_indexes, min_mean_test_score, window_size))
        
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if not os.path.exists(folder + 'Scatter_AHA_CPI_Home-AHA.png'):

    print(' ----- PLOTTING CORRELATIONS ----- ')
    iterations_folders = []

    for iteration in range(number_of_iterations):

        iterations_folders.append(folder + 'Iteration_' + str(iteration) + '/')

    plot_corrcoeff(iterations_folders=iterations_folders, save_folder=folder)

print(' ----- ESECUZIONE DEL MAIN TERMINATA ----- ')


"""
1. Allenare classificatori                                      (train-AHA)

2. Allenare regessore                                           (train-week)

3. Testare miglior classificatore (corfcoef) e regressore (r2)  (test-week)
    4.1 Predict samples sui best estimators (est1, est2, est3)  ->  hp_tot_list_list  
    4.2 Corrcoef tra la prima colonna di hp_tot_list_list e y
    4.3 Calcolare r2 tra regressor(hp_tot_list_list) e y

Iterazione -> corrcoef e r2 / [1, 2, 3][55, 60, 68]
Iterazione -> corrcoef e r2 / [4, 5, 6][55, 60, 68]
Iterazione -> corrcoef e r2 / [7, 8, 9][55, 60, 68]

[1, 2, 34, 4, 5, 6, 7, 8, 9]

best estimators
    est1
    est2
    est3

"""
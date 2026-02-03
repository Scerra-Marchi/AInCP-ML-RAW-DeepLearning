import os
import json
import pandas as pd
from train_regressor import train_regressor
from test_classifier_regressor import test_classifier_regressor

import warnings
import sys

warnings.filterwarnings("ignore")

# Cambio la directory di esecuzione in quella dove si trova questo file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#data_folder = 'C:/Users/david/Documents/University/Borsa di Studio - REDCap/only_AC-80_patients/'
data_folder = '../../AInCP-Training/only_AC-80_patients/'

min_mean_test_score = 0.89 #TODO: change to 0.85
window_size = 300

metadata = pd.read_excel(data_folder + 'metadata2023_08.xlsx')



print(' ----- TRAINING REGRESSORS ----- ')
processes = []

# Flushing the buffers
sys.stdout.flush()
sys.stderr.flush()


save_folder = 'Iterations incriminated 2/Iteration_8/'

# Reading from a JSON file and accessing data
with open(save_folder + 'iteration_data.json', 'r') as file:
    data = json.load(file)
retrieved_train_indexes = data['Train Indexes']
retrieved_test_indexes = data['Test Indexes']

train_regressor(data_folder, save_folder, retrieved_train_indexes, min_mean_test_score, window_size)

print(' ----- TESTING CLASSIFIER AND REGRESSOR ----- ')

test_classifier_regressor(data_folder, save_folder, retrieved_test_indexes, min_mean_test_score, window_size)

# Flushing the buffers
sys.stdout.flush()
sys.stderr.flush()
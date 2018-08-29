# visceral_fat_rating_train.py
# Logistic regression
# CNTK 2.5.1

### IMPORTS:
import numpy as np
import pandas as pd
import cntk as C

### CONSTANT VARIABLES:

### GLOBAL VARIABLES:

### FUNCTION DEFINITIONS:
def print_out_CNTK_version():
    ver = C.__version__
    print(' ### (Using CNTK version ' + str(ver) + ')')

def headers():
    print('\n ### Begin logistic regression on visceral-fat-rating data training')
    print_out_CNTK_version()

def load_data(file_name):
    print('\n Loading data from ' + file_name)

    data_matrix = np.genfromtxt(file_name, dtype = np.float32, delimiter = ',', usecols = range(1, 15))
    
    print(' (Number of rows: ' + str(data_matrix.shape[0]) + ')')
    print(' (Number of columns: ' + str(data_matrix.shape[1]) + ')')

    return data_matrix

def check_for_NaN(data_matrix):
    hasNaN = False
    for i in range(data_matrix.shape[0]):
        for j in range(data_matrix.shape[1]):
            if np.isnan(data_matrix[i, j]):
                hasNaN = True

    # Erase row if data matrix has NaN values in it
    #if(hasNaN):
        #data_matrix = data_matrix[~np.isnan(data_matrix).any(axis=1)]

    return data_matrix

def sort_data_by_column(data_matrix, column):
    data_matrix = data_matrix[data_matrix[:, column].argsort()]

    return data_matrix

def save_data(data_matrix, new_file_name):
    np.savetxt(new_file_name, data_matrix, delimiter=',', newline='\n')

def main():

    data_matrix = load_data('.\\visceral-fat-rating.data')
    checked_data_matrix = check_for_NaN(data_matrix)
    sorted_data_matrix = sort_data_by_column(checked_data_matrix, 13)
    #save_data(sorted_data_matrix, 'sorted_visceral-fat-rating.data')

    ###
    features_matrix = sorted_data_matrix[:, 0:13]
    labels_matrix = np.reshape(sorted_data_matrix[:, 13], (-1, 1))

    ##########################
    print('\n ### End training\n')

### This only executes when 'vesceral_fat_rating_train.py' is executed rather than imported
if __name__ == '__main__':
    headers()
    main()
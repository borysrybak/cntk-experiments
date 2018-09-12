# visceral_fat_rating_train.py
# Logistic regression
# CNTK 2.5.1

### IMPORTS:
import numpy as np
import pandas as pd
import cntk as C
from sklearn import preprocessing

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

def linear_layer(X, features_dimension, labels_dimension):
    W = C.parameter(shape = (features_dimension, 1))
    b = C.parameter(shape = (labels_dimension))

    return C.times(X, W) + b, W, b

def main():

    data_matrix = load_data('.\\visceral-fat-rating.data')
    checked_data_matrix = check_for_NaN(data_matrix)
    sorted_data_matrix = sort_data_by_column(checked_data_matrix, 0)
    #save_data(sorted_data_matrix, 'sorted_visceral-fat-rating.data')

    # features matrix
    unnorm_features_matrix = sorted_data_matrix[:, 1:14]
    min_max_scaler = preprocessing.MinMaxScaler()
    features_matrix = min_max_scaler.fit_transform(unnorm_features_matrix)

    # labels matrix
    uncoded_labels_matrix = np.reshape(sorted_data_matrix[:, 0], (-1, 1))
    labels_logic_matrix = uncoded_labels_matrix > 1
    labels_matrix = labels_logic_matrix.astype(np.float32)
    
    print(' Training data:')
    combined_matrix = np.concatenate((features_matrix, labels_matrix), axis = 1)
    print(combined_matrix)

    features_dimension = 13
    labels_dimension = 1

    X = C.input_variable(features_dimension, np.float32)
    y = C.input_variable(labels_dimension, np.float32)

    z, W, b = linear_layer(X, features_dimension, labels_dimension)
    p = 1.0 / (1.0 + C.exp(-z))

    model = p   

    ###
    cee = C.binary_cross_entropy(model, y)
    eval_error = C.classification_error(model, y)
    learning_rate = 0.01
    learner = C.sgd(model.parameters, learning_rate)

    ###
    trainer = C.Trainer(model, (cee, eval_error), [learner])
    max_iterations = 5000

    ###
    np.random.seed(4)
    N = len(features_matrix)

    for i in range(0, max_iterations):
        row = np.random.choice(N, 1)
        trainer.train_minibatch({
            X: features_matrix[row],
            y: labels_matrix[row]})
        
        if i % 1000 == 0 and i > 0:
            mcee = trainer.previous_minibatch_loss_average
            print(str(i) + ' Cross entropy error on current item = %0.4f ' %mcee)

    # print out results - weights and bias
    np.set_printoptions(precision=4, suppress=True)
    print('Model weights:')
    print(W.value)
    print('Model bias:')
    print(b.value)

    # save results
    print('\nSaving files:')
    weights_file_name = str(learning_rate) + '-' + str(max_iterations) + '_' + 'weights' + '.txt'
    bias_file_name = str(learning_rate) + '-' + str(max_iterations) + '_' + 'bias' + '.txt'

    print(weights_file_name)
    print(bias_file_name)

    np.savetxt(weights_file_name, W.value)
    np.savetxt(bias_file_name, b.value)
    print('Saving complete')

    ##########################
    print('\n ### End training\n')

### This only executes when 'vesceral_fat_rating_train.py' is executed rather than imported
if __name__ == '__main__':
    headers()
    main()
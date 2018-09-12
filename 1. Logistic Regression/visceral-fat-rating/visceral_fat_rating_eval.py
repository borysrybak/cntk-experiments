# visceral_fat_rating_eval.py
# logistic regression visceral-fat-rating data
# model evaluation

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

def loadvaluesfromfile(learning_rate, iteration_number):

    weights_file = '.\\' + str(learning_rate) + '-' + str(iteration_number) + '_weights' + '.txt'
    bias_file =  '.\\' + str(learning_rate) + '-' + str(iteration_number) + '_bias' + '.txt'

    weights_array = np.loadtxt(weights_file, dtype=np.float32, ndmin=2)
    bias_array = np.loadtxt(bias_file, dtype=np.float32)

    weights = np.array(weights_array, dtype=np.float32)
    bias = np.array(bias_array, dtype=np.float32)

    return weights, bias

def compute_p(x, w, b):
    z = 0.0

    for i in range(len(x)):
        z += x[i] * w[i]

    z += b
    p = 1.0 / (1.0 + np.exp(-z))

    return p

def main():
    print('\n Begin logistic regression model evaluation \n')

    # LOADING DATA
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

    # setting weights and bias values
    print('Setting weights and bias values \n')

    learning_rate = 0.01
    max_iterations = 5000
    weights, bias = loadvaluesfromfile(learning_rate, max_iterations)

    N = len(features_matrix)
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    print('item\tpred_prob\tpred_label\tact_label\tresult')

    for i in range(0, N):
        x = features_matrix[i]
        pred_prob = compute_p(x, weights, bias)

        pred_label = ''
        if pred_prob < 0.5:
            pred_label = 0
        else:
            pred_label = 1

        act_label = int(labels_matrix[i])

        pred_str = 'correct' if np.absolute(pred_label - act_label) <= 0 else 'WRONG'
        
        if act_label == 1 & act_label == pred_label:
            true_positives += 1
        if act_label == 1 & act_label != pred_label:
            false_negatives += 1
        if act_label == 0 & act_label == pred_label:
            true_negatives += 1
        if act_label == 0 & act_label != pred_label:
            false_positives += 1
        
        print('%2d\t%0.4f\t\t%0.0f\t\t%0.0f\t\t%s' %(i, pred_prob, pred_label, act_label, pred_str))

    # * True positives are how often your model correctly predicted a tumour was a class 4 (malignant)
    # * False positives are how often your model predicted a tumour was a class 4 (malignant) when it was a class 2 (benign)(i.e your model predicted incorrectly)
    # * True negatives indicate how often your model correctly predicted a tumour was a class 2 (benign)
    # * False negatives indicate how often your model predicted a tumour was class 2 (benign) when in fact it was class 4 (malignant) (i.e. your model predicted incorrectly)

    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    print('\n')
    acc_result = 'Accuracy: ' + str(accuracy) + '\n'
    pre_result = 'Precision: ' + str(precision) + '\n'
    rec_result = 'Recall: ' + str(recall) + '\n'
    print(acc_result)
    print(pre_result)
    print(rec_result)
    results = acc_result + pre_result + rec_result
    results_file_name = 'results_for_LR' + str(learning_rate) + '_' + 'MAX-ITER' + str(max_iterations) + '.txt'

    file = open(results_file_name, 'w')
    file.write(results)
    file.close()

    print('\n End evaluation\n')

### This only executes when 'vesceral_fat_rating_eval.py' is executed rather than imported
if __name__ == '__main__':
    headers()
    main()
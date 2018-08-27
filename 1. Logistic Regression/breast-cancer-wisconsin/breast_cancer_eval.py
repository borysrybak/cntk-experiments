# breast_cancer_eval.py
# logistic regression breast-cancer wisconsin data
# model evaluation

import numpy as np

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
    data_file = '.\\sorted-breast-cancer-wisconsin.data'
    print('\nLoading data from ' + data_file + '\n')

    data_matrix = np.genfromtxt(data_file, dtype=np.float32, delimiter=',', usecols=range(0,10))
    
    # features matrix
    features_matrix = data_matrix[:, 0:9]

    # labels matrix
    unshaped_labels_matrix = data_matrix[:, 9]
    labels_matrix = np.reshape(unshaped_labels_matrix, (-1, 1))

    # setting weights and bias values
    print('Setting weights and bias values \n')

    learning_rate = 0.001
    max_iterations = 10000
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
            pred_label = 2
        else:
            pred_label = 4

        act_label = int(labels_matrix[i])

        pred_str = 'correct' if np.absolute(pred_label - act_label) <= 0 else 'WRONG'
        
        if act_label == 4 & act_label == pred_label:
            true_positives += 1
        if act_label == 4 & act_label != pred_label:
            false_negatives += 1
        if act_label == 2 & act_label == pred_label:
            true_negatives += 1
        if act_label == 2 & act_label != pred_label:
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

    print('\n End evaliation \n')

if __name__ == '__main__':
    main()
# breast_cancer_train.py
# logisitc regression breast-cancer wisconsin data
# CNTK 2.5.1

import numpy as np
import cntk as C
import math
import pandas as pd

def main():
    # HEADERS
    print('\n Begin logistic regression on breast-cancer-wisconsin data training')
    ver = C.__version__
    print('(Using CNTK version ' + str(ver) + ')')

    # LOADING DATA
    data_file = '.\\breast-cancer-wisconsin.data'
    print('\nLoading data from ' + data_file + '\n')

    data_matrix = np.genfromtxt(data_file, dtype=np.float32, delimiter=',', usecols=range(1,11))

    # checking for NaNs and filtering data
    for i in range(699):
        for j in range(10):
            if np.isnan(data_matrix[i,j]):
                location = str(i) + ', ' + str(j)

    filtered_data_matrix = data_matrix[~np.isnan(data_matrix).any(axis=1)]
    sorted_by_label_data_matrix = filtered_data_matrix[filtered_data_matrix[:,9].argsort()]
    np.savetxt('sorted-breast-cancer-wisconsin.data', sorted_by_label_data_matrix, delimiter=',', newline='\n')

    # features matrix
    features_matrix = sorted_by_label_data_matrix[:, 0:9]
    #print(features_matrix)

    # labels matrix - sorted and encoded to 0 or 1
    unshaped_labels_matrix = sorted_by_label_data_matrix[:, 9]
    uncoded_labels_matrix = np.reshape(unshaped_labels_matrix, (-1, 1))
    labels_logic_matrix = uncoded_labels_matrix > 2
    labels_matrix = labels_logic_matrix.astype(np.float32)
    #print(labels_logic_matrix)
    #print(labels_matrix)
    #print(labels_matrix.shape)

    # making training data
    print('Training data:')
    combined_matrix = np.concatenate((features_matrix, labels_matrix), axis=1)
    #print(combined_matrix)

    # create a model
    features_dimension = 9 # x1, x2, x3, x4, x5, x6, x7, x8, x9
    labels_dimension = 1 # always 1 for logistic regression, y

    X = C.input_variable(features_dimension, np.float32)    # cntk.Variable
    y = C.input_variable(labels_dimension, np.float32)      # correct class value

    W = C.parameter(shape=(features_dimension, 1))          # trainable cntk.Parameter
    b = C.parameter(shape=(labels_dimension))

    z = C.times(X, W) + b                                   # or z = C.plus(C.times(X, W), b)
    p = 1.0 / (1.0 + C.exp(-z))                             # or p = C.sigmoid(z)

    model = p                                               # create 'model' alias

    # create learner
    cross_entropy_error = C.binary_cross_entropy(model, y)
    learning_rate = 0.001
    learner = C.sgd(model.parameters, learning_rate)

    # create trainer
    trainer = C.Trainer(model, (cross_entropy_error), [learner])
    max_iterations = 10000

    # train
    print('Start training')
    print('Iterations: ' + str(max_iterations))
    print('Learning Rate (LR): ' + str(learning_rate))
    print('Mini-batch = 1')

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

    print('Training complete')

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

    print('\n End training\n')

if __name__ == '__main__':
    main()
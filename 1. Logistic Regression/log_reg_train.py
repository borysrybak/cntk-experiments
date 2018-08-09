# log_reg_train.py
# logisitc regression age-education-sex synthetic data
# CNTK 2.5.1

import numpy as np
import cntk as C

def main():
    print('\nBegin logistic regression training demo')
    ver = C.__version__
    print('(Using CNTK version ' + str(ver) + ')')

    # training data format:
    # 4.0, 3.0, 1
    # 9.0, 5.0, 1
    # . . .

    data_file = '.\\age_edu_sex.txt'
    print('\nLoading data from ' + data_file + '\n')

    features_matrix = np.loadtxt(data_file, dtype=np.float32, delimiter=',',
        skiprows=0, usecols=[0,1])
    # print(features_matrix)

    labels_matrix = np.loadtxt(data_file, dtype=np.float32, delimiter=',',
        skiprows=0, usecols=[2], ndmin=2)
    # print(labels_matrix)
    # print(labels_matrix.shape)

    print('Training data:')
    combined_matrix = np.concatenate((features_matrix, labels_matrix), axis=1)
    print(combined_matrix)

    # create model
    features_dimension = 2 # x1, x2
    labels_dimension = 1 # always 1 for logistic regression

    X = C.input_variable(features_dimension, np.float32)    # cntk.Variable
    y = C.input_variable(labels_dimension, np.float32)      # correct class value

    W = C.parameter(shape=(features_dimension, 1))  # trainable cntk.Parameter
    b = C.parameter(shape=(labels_dimension))

    z = C.times(X, W) + b           # or z = C.plus(C.times(X, W), b)
    p = 1.0 / (1.0 + C.exp(-z))     # or p = C.sigmoid(z)

    model = p   # create an alias

    # create Learner and Trainer
    cross_entropy_error = C.binary_cross_entropy(model, y) # Cross entropy a bit more principled for Learning Rate
    # squared_error = C.squared_error(model, y)
    learning_rate = 0.010
    learner = C.sgd(model.parameters, learning_rate) # stochastic gradient descent, adadelta, adam, nesterov
    trainer = C.Trainer(model, (cross_entropy_error), [learner])
    max_iterations = 4000

    # train
    print('Start training')
    print('Iterations: ' + str(max_iterations))
    print('Learning Rate (LR): ' + str(learning_rate))
    print('Mini-batch = 1')

    np.random.seed(4)
    N = len(features_matrix)

    for i in range(0, max_iterations):
        row = np.random.choice(N, 1)    # pick a random row from training items
        trainer.train_minibatch({
            X: features_matrix[row],
            y: labels_matrix[row]})
        
        if i % 1000 == 0 and i > 0:
            mcee = trainer.previous_minibatch_loss_average
            print(str(i) + ' Cross entropy error on current item = %0.4f ' %mcee)
    
    print('Training complete')

    # print out results
    np.set_printoptions(precision=4, suppress=True)
    print('Model weights:')
    print(W.value)
    print('Model bias:')
    print(b.value)

if __name__ == '__main__':
    main()
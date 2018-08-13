# log_reg_eval.py
# logistic regression age-education-sex data

import numpy as np

def compute_p(x, w, b):
    z = 0.0

    for i in range(len(x)):
        z += x[i] * w[i]

    z += b
    p = 1.0 / (1.0 + np.exp(-z))

    return p

def main():
    print('\n Begin logistic regression model evaluation \n')

    data_file = '.\\age_edu_sex.txt'
    print('Loading data from ' + data_file)

    features_matrix = np.loadtxt(data_file, dtype=np.float32, delimiter=',',
        skiprows=0, usecols=(0,1))
    labels_matrix = np.loadtxt(data_file, dtype=np.float32, delimiter=',',
        skiprows=0, usecols=[2], ndmin=2)

    print('Setting weights and bias values \n')
    weights = np.array([-0.2049, 0.9666], dtype=np.float32)
    bias = np.array([-2.2864], dtype=np.float32)

    N = len(features_matrix)
    features_dimension = 2

    print('item\tpred_prob\tpred_label\tact_label\tresult')
    
    for i in range(0, N):
        x = features_matrix[i]
        pred_prob = compute_p(x, weights, bias)
        pred_label = 0 if pred_prob < 0.5 else 1
        act_label = labels_matrix[i]
        pred_str = 'correct' if np.absolute(pred_label - act_label) < 1.0e-5 \
            else 'WRONG'
        print('%2d\t%0.4f\t\t%0.0f\t\t%0.0f\t\t%s' % \
            (i, pred_prob, pred_label, act_label, pred_str))

    x = np.array([9.5, 4.5], dtype=np.float32)
    print('\nPredicting class for age, education = ')
    print(x)

    p = compute_p(x, weights, bias)
    print('Predicted p = ' + str(p))
    if p < 0.5: print('Predicted class = 0')
    else: print('Predicted class = 1')

    print('\n End evaluation\n')

if __name__ == '__main__':
    main()
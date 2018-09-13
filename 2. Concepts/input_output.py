# input_output.py
# Demo the NN input-output mechanism
# CNTK 2.5.1

### IMPORTS:
import numpy as np
import cntk as C

### CONSTANT VARIABLES:

### GLOBAL VARIABLES:

### FUNCTION DEFINITIONS:
def print_out_CNTK_version():
    ver = C.__version__
    print(' ### (Using CNTK version ' + str(ver) + ')')

def headers():
    print('\n ### Begin neural network input-output demo')
    print_out_CNTK_version()

def main():
    np.set_printoptions(precision = 4, suppress = True, formatter = {'float': '{: 0.2f}'.format})

    input_node_dim = 4
    hidden_node_dim = 2
    output_node_dim = 3

    X = C.input_variable(input_node_dim, np.float32)
    Y = C.input_variable(output_node_dim, np.float32)

    print('\nCreating a 4-2-3 tanh-softmax neural network')
    h = C.layers.Dense(hidden_node_dim, activation = C.tanh, name = 'hidLayer')(X)
    o = C.layers.Dense(output_node_dim, activation = C.softmax, name = 'outLayer')(h)
    nnet = o

    print('\nSetting weights and bias values')
    ih_wts = np.array([[0.01, 0.02],
                       [0.03, 0.04],
                       [0.05, 0.06],
                       [0.07, 0.08]], dtype = np.float32)

    h_biases = np.array([0.09, 0.10])

    ho_wts = np.array([[0.11, 0.12, 0.13],
                       [0.14, 0.15, 0.16]], dtype = np.float32)

    o_biases = np.array([0.17, 0.18, 0.19], dtype = np.float32)

    h.hidLayer.W.value = ih_wts
    h.hidLayer.b.value = h_biases
    o.outLayer.W.value = ho_wts
    o.outLayer.b.value = o_biases

    print('\nSet the input-hidden weights to: ')
    print(h.hidLayer.W.value)
    print('Set the hidden node biases to: ')
    print(h.hidLayer.b.value)
    print('Set the hidden-output weights to: ')
    print(o.outLayer.W.value)
    print('Set the output node biases to: ')
    print(o.outLayer.b.value)

    print('\nSetting input values to (1.0, 2.0, 3.0, 4.0)')
    x_vals = np.array([1.0, 2.0, 3.0, 4.0], dtype = np.float32)

    np.set_printoptions(formatter = {'float': '{: 0.4f}'.format})
    print('\nFeeding input values to hidden layer only ')
    h_vals = h.eval({X: x_vals})
    print('Hidden node values:')
    print(h_vals)

    print('\nFeeding input values to entire network ')
    y_vals = nnet.eval({X: x_vals})
    print('Output node values:')
    print(y_vals)

    ##########################
    print('\n ### End demo\n')

### This only executes when 'input_output.py' is executed rather than imported
if __name__ == '__main__':
    headers()
    main()
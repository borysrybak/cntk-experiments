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
    hidden_layer = C.layers.Dense(hidden_node_dim, activation = C.tanh, name = 'hidLayer')(X)
    output_layer = C.layers.Dense(output_node_dim, activation = C.softmax, name = 'outLayer')(hidden_layer)
    neural_network = output_layer # or neural_network = C.ops.alias(output_layer) 

    print('\nSetting weights and bias values')
    input_to_hidden_weights = np.array([[0.01, 0.02],
                       [0.03, 0.04],
                       [0.05, 0.06],
                       [0.07, 0.08]], dtype = np.float32)

    hidden_biases = np.array([0.09, 0.10])

    hidden_to_output_weights = np.array([[0.11, 0.12, 0.13],
                       [0.14, 0.15, 0.16]], dtype = np.float32)

    output_biases = np.array([0.17, 0.18, 0.19], dtype = np.float32)

    hidden_layer.hidLayer.W.value = input_to_hidden_weights
    hidden_layer.hidLayer.b.value = hidden_biases
    output_layer.outLayer.W.value = hidden_to_output_weights
    output_layer.outLayer.b.value = output_biases

    print('\nSet the input-hidden weights to: ')
    print(hidden_layer.hidLayer.W.value)
    print('Set the hidden node biases to: ')
    print(hidden_layer.hidLayer.b.value)
    print('Set the hidden-output weights to: ')
    print(output_layer.outLayer.W.value)
    print('Set the output node biases to: ')
    print(output_layer.outLayer.b.value)

    print('\nSetting input values to (1.0, 2.0, 3.0, 4.0)')
    input_values = np.array([1.0, 2.0, 3.0, 4.0], dtype = np.float32)

    np.set_printoptions(formatter = {'float': '{: 0.4f}'.format})
    print('\nFeeding input values to hidden layer only ')
    hidden_values = hidden_layer.eval({X: input_values})
    print('Hidden node values:')
    print(hidden_values)

    print('\nFeeding input values to entire network ')
    output_values = neural_network.eval({X: input_values})
    print('Output node values:')
    print(output_values)

    ##########################
    print('\n ### End demo\n')

### This only executes when 'input_output.py' is executed rather than imported
if __name__ == '__main__':
    headers()
    main()
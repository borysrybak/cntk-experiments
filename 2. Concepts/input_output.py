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


    ##########################
    print('\n ### End demo\n')

### This only executes when 'input_output.py' is executed rather than imported
if __name__ == '__main__':
    headers()
    main()
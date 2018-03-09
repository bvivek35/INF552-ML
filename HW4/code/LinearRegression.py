#!/usr/bin/env python3

'''
    Class: INF552 at USC
    Submission Member:
                        Vivek Bharadwaj <vivekb> <vivekb@usc.edu>
                        Shushyam Malige Sharanappa <maligesh> <maligesh@usc.edu>
                        Raveena Mathur <raveenam> <raveenam@usc.edu>
    This is a python implementation of Linear Regression.
'''

__version__ = '1.0'

# Imports
import numpy as np
# End imports

class LinearRegression():
    '''
        Implements Linear Regression.
    '''
    def __init__(self, weights=[]):
        self.weights = weights
    
    def train(self, X, Y):
        '''
            X = Data Matrix.
            But in Formula, 

            W = (D * D.T)^-1 * D * Y,
            D is of order (d X N)
            But X is of order (N X d)
        '''
        D = X.T
        tmp = np.linalg.inv(np.matmul(D, D.T))
        self.weights = np.matmul(np.matmul(tmp, D), Y)

if __name__ == '__main__':
    import sys

    HELP_TEXT = 'USAGE: {0} <input file>'
    if len(sys.argv) != 2:
        print(HELP_TEXT.format(sys.argv[0]))
        sys.exit(1)
    else:
        inFile = sys.argv[1]
    
    data = np.loadtxt(inFile, \
                    delimiter=',', \
                    dtype='float', \
                    usecols=(0,1,2) \
                )

    X = data[:, :-1]
    Y = data[:, -1]

    model = LinearRegression()
    model.train(X, Y)

    print('Final Weights: {0}'.format(model.weights))
#!/usr/bin/env python3

'''
    Class: INF552 at USC
    Submission Member:
                        Vivek Bharadwaj <vivekb> <vivekb@usc.edu>
                        Shushyam Malige Sharanappa <maligesh> <maligesh@usc.edu>
                        Raveena Mathur <raveenam> <raveenam@usc.edu>
    This is a python implementation of Logistic Regression.
'''

__version__ = '1.0'

# Imports
import numpy as np
import math
# End imports

class LogisticRegression():
    '''
        Implements Logistic Regression.
    '''

    @staticmethod
    def findGradient(weights, x, y):
        arg = y * np.dot(weights, x)
        tmp = 1 + np.exp(arg)
        res = (x * y) / tmp
        return res

    def __init__(self, weights=[], alpha=0.001, maxIter=1000):
        self.weights = weights
        self.alpha = alpha
        self.maxIter = maxIter
    
    def train(self, X, Y):
        N, d = X.shape
        X = np.insert(X, 0, 1, axis=1)
        self.weights = np.random.random(d+1)
        iter = 0
        while iter < self.maxIter:
            gradient = np.zeros(d+1)
            for x, y in zip(X, Y):
                gradient = np.add(gradient, LogisticRegression.findGradient(self.weights, x, y))
            gradient /= N
            self.weights += self.alpha * gradient
            iter += 1

if __name__ == '__main__':
    import sys

    HELP_TEXT = 'USAGE: {0} <input file>'
    if len(sys.argv) != 2:
        print(HELP_TEXT.format(sys.argv[0]))
    else:
        inFile = sys.argv[1]
    
    data = np.loadtxt(inFile, \
                    delimiter=',', \
                    dtype='float', \
                    usecols=(0,1,2,4) \
                )

    X = data[:, :-1]
    Y = data[:, -1]

    model = LogisticRegression(alpha=0.05, maxIter=7000)
    model.train(X, Y)
    print('Final Weights: {0}'.format(model.weights))
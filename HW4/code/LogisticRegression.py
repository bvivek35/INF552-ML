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
    def sigmoid(arg):
        tmp = np.exp(arg)
        # tmp1 = math.exp(arg)
        return 1 / (1 + tmp)

    @staticmethod
    def sigmoidArg(w, x, y):
        return  - y * np.dot(w.T, x)

    @staticmethod
    def gradient(weights, x, y, N):
        sigArg = LogisticRegression.sigmoidArg(weights, x, y)
        print('sigarg: {0}, {1}, {2}, {3}'.format(weights, x, y, sigArg))
        sig = LogisticRegression.sigmoid(sigArg)
        expPart = np.exp(sigArg)
        tmp = -(sig * expPart * y * x) / N
        # tmp = -(sig * y * x)
        return tmp

    def __init__(self, weights=[], alpha=0.001, maxIter=1000):
        self.weights = weights
        self.alpha = alpha
        self.maxIter = maxIter
    
    def train(self, X, Y):
        N, d = X.shape
        self.weights = np.random.random(d)
        iter = 0
        while iter < self.maxIter:
            tmp = np.zeros(d)
            for x, y in zip(X, Y):
                tmp += LogisticRegression.gradient(self.weights, x, y, N)
            self.weights -= self.alpha * tmp
            iter += 1

    def predict(self, X):
        Y = []
        for x in X:
            arg_neg = LogisticRegression.sigmoidArg(self.weights, x, -1)
            prob_neg = LogisticRegression.sigmoid(arg_neg)
            arg_pos = LogisticRegression.sigmoidArg(self.weights, x, 1)
            prob_pos = LogisticRegression.sigmoid(arg_pos)
            if prob_neg > prob_pos:
                res = -1
            else:
                res = 1
            Y.append(res)
        
        return Y

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

    model = LogisticRegression(alpha=0.001, maxIter=2000)
    model.train(X, Y)
    model.predict(X)
    print('Final Weights: {0}'.format(model.weights))
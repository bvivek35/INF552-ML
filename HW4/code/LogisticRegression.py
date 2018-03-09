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
        '''
            Implements the below:
            Nr = Yi * Xi
            Dr = 1 + exp(Yi * wT dot Xi)
            return Nr / Dr
        '''
        arg = y * np.dot(weights, x)
        tmp = 1 + np.exp(arg)
        res = (x * y) / tmp
        return res
    
    @staticmethod
    def findProb(weights, x, y):
        arg = y * np.dot(weights, x)
        tmp = np.exp(arg)
        return tmp / (1 + tmp)


    def __init__(self, weights=[], alpha=0.001, maxIter=1000):
        self.weights = weights
        self.alpha = alpha
        self.maxIter = maxIter
    
    def train(self, X, Y, verbose=False):
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

            if verbose:
                if iter % 500 == 0:
                    print('Completed iterations: {0}'.format(iter))

        return iter

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        Y = [None for _ in range(X.shape[0])]
        for idx, x in enumerate(X):
            prob_1 = LogisticRegression.findProb(self.weights, x, 1)
            if prob_1 > 0.5:
                Y[idx] = 1
            else:
                Y[idx] = -1

        return np.asarray(Y)


if __name__ == '__main__':
    import sys

    HELP_TEXT = 'USAGE: {0} <learning-rate/alpha> <max no. of Iterations>  <input file>'
    if len(sys.argv) != 4:
        print(HELP_TEXT.format(sys.argv[0]))
        sys.exit(1)
    else:
        alpha = float(sys.argv[1])
        maxIter = int(sys.argv[2])
        inFile = sys.argv[3]
    
    data = np.loadtxt(inFile, \
                    delimiter=',', \
                    dtype='float', \
                    usecols=(0,1,2,4) \
                )

    X = data[:, :-1]
    Y = data[:, -1]

    model = LogisticRegression(alpha=alpha, maxIter=maxIter)
    nIterations = model.train(X, Y, verbose=True)

    YPred = model.predict(X)
    nCorrect = np.where(Y==YPred)[0].shape[0]
    nTotal = YPred.shape[0]
    accuracy = nCorrect / nTotal

    print('No. of Iterations: {0}'.format(nIterations))
    print('Final Weights [W0, W1, W2,...]: {0}'.format(model.weights))
    print('\n\t\t'.join([
            'Accuracy on the train dataset: {0}', 
            'Predicted Correctly: {1}', 
            'Total Samples: {2}'
            ]).format(accuracy, nCorrect, nTotal)
        )

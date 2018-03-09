#!/usr/bin/env python3

'''
    Class: INF552 at USC
    Submission Member:
                        Vivek Bharadwaj <vivekb> <vivekb@usc.edu>
                        Shushyam Malige Sharanappa <maligesh> <maligesh@usc.edu>
                        Raveena Mathur <raveenam> <raveenam@usc.edu>
    This is a python implementation of the Perceptron Learning Algorithm.
'''

__version__ = '1.0'

# Imports
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
# End imports

class Perceptron():
    '''
        Implements the Perceptron Learning Algorithm.
    '''
    def __init__(self, weights=[], alpha=0.01, maxIter=1000):
        self.weights = weights
        self.alpha = alpha
        self.maxIter = maxIter
        self.errorCounts = []
    
    def train(self, X, Y, verbose=False):
        d = X.shape[1]
        X = np.insert(X, 0, 1, axis=1)
        self.weights = np.random.random(d+1) # make space for W0
        iter = 0
        while iter < self.maxIter:
            self.errorCounts.append(0)
            for x, y in zip(X, Y):
                prod = np.dot(x, self.weights)
                if prod > 0 and y < 0:
                    self.weights -= self.alpha * x
                elif prod < 0 and y > 0:
                    self.weights += self.alpha * x

            tmp = np.sign(np.dot(X, self.weights))
            self.errorCounts[-1] = X.shape[0] - np.where(tmp==Y)[0].shape[0]
            iter += 1
            if self.errorCounts[-1] == 0:
                break
            
            if verbose:
                if iter % 500 == 0:
                    print('Completed iterations: {0}'.format(iter))


        return iter

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return np.sign(np.dot(X, self.weights))


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
                    usecols=(0,1,2,3) \
                )

    X = data[:, :-1]
    Y = data[:, -1]

    model = Perceptron(alpha=alpha, maxIter=maxIter)
    nIterations = model.train(X, Y, verbose=True)

    YPred = model.predict(X)
    nCorrect = np.where(Y==YPred)[0].shape[0]
    nTotal = YPred.shape[0]
    accuracy = nCorrect / nTotal

    print('No. of Iterations: {0}'.format(nIterations))
    print('Final Weights [W0, W1, W2,...]: {0}'.format(model.weights))
    print('Final No. of Violations: {0}'.format(model.errorCounts[-1]))
    print('\n\t\t'.join([
            'Accuracy on the train dataset: {0}', 
            'Predicted Correctly: {1}', 
            'Total Samples: {2}'
            ]).format(accuracy, nCorrect, nTotal)
        )
    '''
    # Acc to new question, no need to plot this.

    plt.ylabel('No. of violations')
    plt.xlabel('iteration')
    plt.plot(model.errorCounts)
    plt.show()
    '''
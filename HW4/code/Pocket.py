#!/usr/bin/env python3

'''
    Class: INF552 at USC
    Submission Member:
                        Vivek Bharadwaj <vivekb> <vivekb@usc.edu>
                        Shushyam Malige Sharanappa <maligesh> <maligesh@usc.edu>
                        Raveena Mathur <raveenam> <raveenam@usc.edu>
    This is a python implementation of Pocket Algorithm - an adaptation of the Perceptron Learning for linearly non-seperable data points.
'''

__version__ = '1.0'

# Imports
from collections import namedtuple
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# End imports

class Perceptron():
    '''
        Pocket Algorithm - an adaptation of the Perceptron Learning.
    '''
    def __init__(self, weights=[], alpha=0.01):
        self.alpha = alpha
        self.weights = weights
        self.errorCounts = []
        self.bestWeights = []
        self.bestErrorCount = float('inf')
    
    def train(self, X, Y, maxIter=1000):
        d = X.shape[1]
        X = np.insert(X, 0, 1, axis=1)
        self.weights = np.random.random(d+1) # make space for W0
        iter = 0
        while iter < maxIter:
            self.errorCounts.append(0)
            for x, y in zip(X, Y):
                prod = np.dot(x, self.weights)
                if prod > 0 and y < 0:
                    self.weights -= self.alpha * x
                elif prod < 0 and y > 0:
                    self.weights += self.alpha * x

            tmp = np.sign(np.dot(X, self.weights))
            self.errorCounts[-1] = X.shape[0] - np.where(tmp==Y)[0].shape[0]
            if self.bestErrorCount > self.errorCounts[-1]:
                self.bestErrorCount = self.errorCounts[-1]
                self.bestWeights = copy.deepcopy(self.weights)
                self.bestIterationNo = iter
            iter += 1
            if self.errorCounts[-1] == 0:
                break

        return iter

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

    model = Perceptron(alpha=0.01)
    nIterations = model.train(X, Y, maxIter=7000)

    print('No. of Iterations: {0}'.format(nIterations))
    print('Best Weights: {0}'.format(model.bestWeights))
    print('Best/Least No. of Violations: {0}'.format(model.bestErrorCount))
    print('Iteration of occurence: {0}'.format(model.bestIterationNo))
    
    plt.ylabel('No. of violations')
    plt.xlabel('iteration')
    plt.plot(model.errorCounts)
    plt.show()
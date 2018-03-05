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
    
    def train(self, X, Y):
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
                    usecols=(0,1,2,3) \
                )

    X = data[:, :-1]
    Y = data[:, -1]

    model = Perceptron(alpha=0.01, maxIter=3000)
    nIterations = model.train(X, Y)

    print('No. of Iterations: {0}'.format(nIterations))
    print('Final Weights: {0}'.format(model.weights))
    print('Final No. of Violations: {0}'.format(model.errorCounts[-1]))
    
    plt.ylabel('No. of violations')
    plt.xlabel('iteration')
    plt.plot(model.errorCounts)
    plt.show()
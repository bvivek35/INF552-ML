#!/usr/bin/env python3

'''
    Class: INF552 at USC
    Submission Member: 
                        Vivek Bharadwaj <vivekb> <vivekb@usc.edu>
                        Shushyam Malige Sharanappa <maligesh> <maligesh@usc.edu>
                        Raveena Mathur <raveenam> <raveenam@usc.edu>
    This is a python implementation of the Perceptron Learning Algorithm using the Scikit-Learn Library.
'''

__version__ = '1.0'

if __name__ == '__main__':
    # Imports
    import sys
    from numpy import loadtxt
    from sklearn.linear_model import Perceptron
    # End imports

    HELP_TEXT = 'USAGE: {0} <input file>'
    if len(sys.argv) != 2:
        print(HELP_TEXT.format(sys.argv[0]))
    else:
        inFile = sys.argv[1]
    
    data = loadtxt(inFile, \
                    delimiter=',', \
                    dtype='float', \
                    usecols=(0,1,2,3) \
                )

    X = data[:, :-1]
    Y = data[:, -1]

    model = Perceptron(alpha=0.000001, n_iter=3000, verbose=False)
    model.fit(X, Y)
    print(model.coef_)
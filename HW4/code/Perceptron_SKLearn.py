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
    from numpy import loadtxt, where
    from sklearn.linear_model import Perceptron
    # End imports

    HELP_TEXT = 'USAGE: {0} <learning-rate/alpha> <max no. of Iterations>  <input file>'
    if len(sys.argv) != 4:
        print(HELP_TEXT.format(sys.argv[0]))
        sys.exit(1)
    else:
        alpha = float(sys.argv[1])
        maxIter = int(sys.argv[2])
        inFile = sys.argv[3]

    data = loadtxt(inFile, \
                    delimiter=',', \
                    dtype='float', \
                    usecols=(0,1,2,3) \
                )

    X = data[:, :-1]
    Y = data[:, -1]

    model = Perceptron(alpha=alpha, n_iter=maxIter, verbose=False)
    model.fit(X, Y)
    YPred = model.predict(X)

    nCorrect = where(Y==YPred)[0].shape[0]
    nTotal = YPred.shape[0]
    accuracy = nCorrect / nTotal


    print('Final Weights: {0}'.format(model.coef_))
    print('\n\t\t'.join([
        'Accuracy on the train dataset: {0}', 
        'Predicted Correctly: {1}', 
        'Total Samples: {2}'
        ]).format(accuracy, nCorrect, nTotal)
    )

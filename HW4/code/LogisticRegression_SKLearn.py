#!/usr/bin/env python3

'''
    Class: INF552 at USC
    Submission Member:
                        Vivek Bharadwaj <vivekb> <vivekb@usc.edu>
                        Shushyam Malige Sharanappa <maligesh> <maligesh@usc.edu>
                        Raveena Mathur <raveenam> <raveenam@usc.edu>
    This is a python implementation of Logistic Regression using Scikit-Learn library.
'''

__version__ = '1.0'

if __name__ == '__main__':
    # Imports
    import sys
    from numpy import loadtxt, where
    from sklearn.linear_model import LogisticRegression
    # End imports

    HELP_TEXT = 'USAGE: {0} <input file>'
    if len(sys.argv) != 2:
        print(HELP_TEXT.format(sys.argv[0]))
        sys.exit(1)
    else:
        inFile = sys.argv[1]
    
    data = loadtxt(inFile, \
                    delimiter=',', \
                    dtype='float', \
                    usecols=(0,1,2,4) \
                )

    X = data[:, :-1]
    Y = data[:, -1]

    model = LogisticRegression()
    model.fit(X, Y)
    YPredict = model.predict(X)
    

    print('Final Weights: {0}'.format(model.coef_))
    print('No. of Correct Predict: {0}'.format(where(YPredict == Y)[0].shape))
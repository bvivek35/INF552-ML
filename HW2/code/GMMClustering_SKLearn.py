#!/usr/bin/env python3

'''
    Class: INF552 at USC
    Submission Member:
                        Vivek Bharadwaj <vivekb> <vivekb@usc.edu>
                        Shushyam Malige Sharanappa <maligesh> <maligesh@usc.edu>
                        Raveena Mathur <raveenam> <raveenam@usc.edu>
    This is a python example of Clustering using Gaussian Mixtures from the Scikit-Learn library.
'''

__version__ = '1.0'

if __name__ == '__main__':
    # Imports
    import sys
    from numpy import genfromtxt
    from sklearn.mixture import GaussianMixture
    # End imports

    HELP_TEXT = 'USAGE: {0} <ClustersDataCSVFile> <NumberOfClusters>'
    if len(sys.argv) != 3:
        print(HELP_TEXT.format(sys.argv[0]))
        sys.exit(1)
    else:
        inFile = sys.argv[1]
        NClusters = int(sys.argv[2])

    X = genfromtxt(inFile, delimiter = ',')

    gmm = GaussianMixture(n_components=3)
    gmm.fit(X)
    for idx, (mean, covar, amp) in enumerate(zip(gmm.means_, gmm.covariances_, gmm.weights_)):
        print('Stats of Gaussian {0}:\nMean:\t\t{1}\nCovariance:\t{2}\nWeight:\t\t{3}'.format(idx, mean, covar, amp))
        print('-' * 40)
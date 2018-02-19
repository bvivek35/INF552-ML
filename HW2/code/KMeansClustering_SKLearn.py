#!/usr/bin/env python3

'''
    Class: INF552 at USC
    Submission Members:
                        Vivek Bharadwaj <vivekb> <vivekb@usc.edu>
                        Shushyam Malige Sharanappa <maligesh> <maligesh@usc.edu>
                        Raveena Mathur <raveenam> <raveenam@usc.edu>
    This is a python implementation of Clustering using K-Means Algorithm from the Scikit-Learn library.
'''

__version__ = '1.0'

from sklearn.cluster import KMeans
import numpy as np

if __name__ == '__main__':
    import sys
    HELP_TEXT = 'USAGE: {0} <ClustersDataCSVFile> <NumberOfClusters>'
    if len(sys.argv) != 3:
        print(HELP_TEXT.format(sys.argv[0]))
        sys.exit(1)
    else:
        inFile = sys.argv[1]
        NClusters = int(sys.argv[2])

    X = np.genfromtxt(inFile, delimiter = ',')
    kmeans = KMeans(n_clusters=NClusters, random_state=0).fit(X)
    for idx, centroid in enumerate(kmeans.cluster_centers_):
        print('Cluster {0} - centroid {1}'.format(idx, centroid))

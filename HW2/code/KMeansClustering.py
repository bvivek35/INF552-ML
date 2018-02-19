#!/usr/bin/env python3

'''
    Class: INF552 at USC
    Submission Member: Vivek Bharadwaj <vivekb> <vivekb@usc.edu>
    This is a python implementation of Clustering using K-Means Algorithm.
    The Algorithm will accept numpy array of array: shape -> (XXXX, 2)
'''

__author__ = 'Vivek Bharadwaj'
__email__ = 'vivekb@usc.edu'
__version__ = '1.0'

# Imports
import math
from collections import namedtuple
import os # For checking PY_USER_LOG environ var for logging
import logging
import pprint
import numpy as np
# End imports

# Setup logging
LOG_LEVEL = logging.getLevelName(os.environ.get('PY_USER_LOG', 'CRITICAL').upper())
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Declare decorators
def logArgsRet(logger=None):
    import pprint
    def logArgsWrapped(fn):
        def loggedFn(*args, **kwargs):
            logMsg = 'Calling {0}({1}, {2})'.format( \
                fn.__name__, \
                pprint.pformat(args), \
                pprint.pformat(kwargs))
            if logger:
                logger.debug(logMsg)
            elif print:
                print(logMsg)
            
            ret = fn(*args, **kwargs)

            logMsg = 'Returning {0}'.format(pprint.pformat(ret))
            if logger:
                logger.debug(logMsg)
            elif print:
                print(logMsg)

            return ret

        return loggedFn
    return logArgsWrapped

# Declare types
Cluster = namedtuple('Cluster', ['clusterNumber', 'centroid', 'pts'])

@logArgsRet(logger)
def getRandomPtsWithinRange(pts, numPoints):
    # mins = [min(col1), min(col2), min(col3), ...]
    mins = np.apply_along_axis(min, arr=pts, axis=0)
    # maxs = [max(col1), max(col2), max(col3), ...])
    maxs = np.apply_along_axis(max, arr=pts, axis=0)
    # ranges = [[min1, min2, min3, ...]
    #           [max1, max2, max3, ...]]
    ranges = np.vstack((mins, maxs))
    # centriods = [[C1x, C1y, C1z, ...]
    #              [C2x, C2y, C2z, ...], ...]
    randPts = np.vstack((
        np.apply_along_axis(lambda x: np.random.uniform(x[0], x[1]), arr=ranges, axis=0)
        for _ in range(numPoints)
        ))

    return randPts

@logArgsRet(logger)
def KMeansCluster(pts, numClusters, maxIter=70):
    '''
    Clusters pts to numClusters clusters. Performs the KMeans core alg maxIter times.
    Use maxIter as the convergence indicator.
    Args:
        pts: data points to cluster
        K: number of clusters to form
        maxIter: no. of iterations of KMeans
    Returns:
        ClusterResult
    '''
    centroids = getRandomPtsWithinRange(pts, numClusters)
    logger.info('Picked Random Centroids: {0}'.format(centroids))

    clusters = [Cluster(idx, centroid, []) for idx, centroid in enumerate(centroids)]

    nIter = 0
    while nIter < maxIter:
        # assign pts to clusters
        for pt in  pts:
            nearestClusterIdx = np.argmin([np.linalg.norm(pt - cl.centroid) for cl in clusters])
            clusters[nearestClusterIdx].pts.append(pt)

        logger.info('Iteration Stats: {0}'.format(nIter))
        for cl in clusters:
            logger.info('Cluster {0} - No. of pts - {1} Centroid - {2}'.format(cl.clusterNumber, len(cl.pts), cl.centroid))

        # recompute. Throws away old clusters!
        clusters = [Cluster(cl.clusterNumber, np.mean(cl.pts, axis=0), []) for cl in clusters]

        nIter += 1

    return clusters

if __name__ == '__main__':
    # Imports
    import sys
    from numpy import genfromtxt
    # End imports

    HELP_TEXT = 'USAGE: {0} <ClustersDataCSVFile> <NumberOfClusters>'
    if len(sys.argv) != 3:
        print(HELP_TEXT.format(sys.argv[0]))
        sys.exit(1)
    else:
        inFile = sys.argv[1]
        NClusters = int(sys.argv[2])
    
    pts = genfromtxt(inFile, delimiter = ',')

    clusters = KMeansCluster(pts, numClusters = NClusters)
    for cl in clusters:
        print('Cluster {0} - centroid {1}'.format(cl.clusterNumber, cl.centroid))

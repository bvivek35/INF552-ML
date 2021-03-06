#!/usr/bin/env python3

'''
    Class: INF552 at USC
    Submission Members:
                        Vivek Bharadwaj <vivekb> <vivekb@usc.edu>
                        Shushyam Malige Sharanappa <maligesh> <maligesh@usc.edu>
                        Raveena Mathur <raveenam> <raveenam@usc.edu>
    This is a python implementation of Clustering using K-Means Algorithm.
'''

__version__ = '1.0'

# Imports
import math
from collections import namedtuple
import os # For checking PY_USER_LOG environ var for logging
import logging
import pprint
import numpy as np
import random
# End imports

# Setup logging
LOG_LEVEL = logging.getLevelName(os.environ.get('PY_USER_LOG', 'CRITICAL').upper())
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Declare decorators
def logArgsRet(logger=None):
    import pprint
    def logArgsRetWrapped(fn):
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
    return logArgsRetWrapped

# Declare types
Cluster = namedtuple('Cluster', ['clusterNumber', 'centroid', 'pts'])

@logArgsRet(logger)
def getRandomPtsWithinRange(pts, numPoints):
    '''
        Pick randomly numPoints from pts.
        These will be the initial random centroids.
    '''
    randPts = np.vstack((pts[idx] for idx in random.sample(range(0, len(pts)), numPoints)))

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
    # End imports

    HELP_TEXT = 'USAGE: {0} <ClustersDataCSVFile> <NumberOfClusters>'
    if len(sys.argv) != 3:
        print(HELP_TEXT.format(sys.argv[0]))
        sys.exit(1)
    else:
        inFile = sys.argv[1]
        NClusters = int(sys.argv[2])
    
    pts = np.genfromtxt(inFile, delimiter = ',')

    clusters = KMeansCluster(pts, numClusters = NClusters, maxIter=100)
    for cl in clusters:
        print('Cluster {0} - centroid {1}'.format(cl.clusterNumber, cl.centroid))

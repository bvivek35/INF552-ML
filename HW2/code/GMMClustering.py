#!/usr/bin/env python3

'''
    Class: INF552 at USC
    Submission Member:
                        Vivek Bharadwaj <vivekb> <vivekb@usc.edu>
                        Shushyam Malige Sharanappa <maligesh> <maligesh@usc.edu>
                        Raveena Mathur <raveenam> <raveenam@usc.edu>
    This is a python implementation of Soft Clustering using Gaussian Mixture Models.
'''

__version__ = '1.0'

# Imports
import math
from collections import namedtuple
import os # For checking PY_USER_LOG environ var for logging
import logging
import pprint
import operator
import functools

import numpy as np
from numpy import linalg as LA
# End imports

# Setup logging
LOG_LEVEL = logging.getLevelName(os.environ.get('PY_USER_LOG', 'CRITICAL'))
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
Gaussian = namedtuple('Gaussian', ['mean', 'covar', 'amp'])
GaussianMixture = namedtuple('GaussianMixture', ['gaussians'])

#@logArgsRet(logger)
def genRandomGaussians(pts, dim, nGaussians):
    randIdxs = np.random.randint(0, nGaussians)
    means = [np.matrix(pts[np.random.randint(0, len(pts))]) for _ in range(nGaussians)]

    covars = [np.identity(dim) for _ in range(nGaussians)]

    amps = [1/nGaussians for _ in range(nGaussians)]

    ret = [Gaussian(mean, covar, amp) for mean, covar, amp in zip(means, covars, amps)]
    return ret

#@logArgsRet(logger)
def evalGauss(gauss, x):
    '''
        Eval the guassian at x
        ie. the prob(x|c)
    '''
    dr = (2 * math.pi * \
            np.abs(LA.det(gauss.covar))
        ) ** 0.5

    expArg = -0.5 * \
            np.matrix((x - gauss.mean)) * \
            LA.inv(gauss.covar) * \
            np.matrix((x - gauss.mean)).T

    nr = np.exp(expArg)

    ret = nr / dr

    return ret[0, 0]

#@logArgsRet(logger)
def GMMCluster(pts, K):
    '''
    Args:
        pts: data points to cluster
        K: number of clusters to form
    Returns:
        ClusterResult
    '''
    gmm = GaussianMixture(genRandomGaussians(pts, pts.shape[1], K))

    nIter = 0
    maxIter = 200

    while nIter < maxIter:
        # Est. step - Expectation
        '''
            R = [
                [r11, r21, r31, ... rn1]
                [r12, r22, r32, ... rn2]
                [r1k, r2k, r3k ... rnk]
            ]
            ric -> responsibility of data point (i) to cluster (c)
        '''
        R = [
                np.asarray([
                    gauss.amp * evalGauss(gauss, pt) / sum(g.amp * evalGauss(g, pt) for g in gmm.gaussians)
                    for pt in pts
                ])
                for gauss in gmm.gaussians
            ]

        # Max. step - maximization
        newAmps = [sum(R[idx]) / len(pts) for idx in range(len(gmm.gaussians))]
        newMeans = [np.average(pts, weights=r, axis=0) for r in R]
        newCovars = [
                        sum(
                            r[ptIdx] * np.matrix(pt - newMeans[idx]).T * np.matrix(pt - newMeans[idx])
                                for ptIdx, pt in enumerate(pts)
                            ) / sum(r)
                        for idx, r in enumerate(R)
                    ]

        gmm = GaussianMixture([Gaussian(*args) for args in zip(newMeans, newCovars, newAmps)])

        logger.info('\t'.join(map(str, (gaussian.amp for gaussian in gmm.gaussians))))
        logger.info('\t'.join(map(str, (gaussian.mean for gaussian in gmm.gaussians))))
        logger.info('\t'.join(map(str, (gaussian.covar for gaussian in gmm.gaussians))))

        nIter += 1

    return gmm

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

    gmm = GMMCluster(pts, K = NClusters)
    for idx, gaussian in enumerate(gmm.gaussians):
        print('Gaussian {0}:\n\tMean: {1}\n\tAmplitude: {2}\n\tCovariance: {3}'
                .format(idx, gaussian.mean, gaussian.amp, gaussian.covar))
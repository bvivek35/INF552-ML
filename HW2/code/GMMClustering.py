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
# End imports

# Setup logging
LOG_LEVEL = logging.getLevelName(os.environ.get('PY_USER_LOG', 'CRITICAL'))
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Declare decorators
def logArgs(logger=None):
    import pprint
    def logArgsWrapped(fn):
        def loggedFn(*args, **kwargs):
            logMsg = 'Calling {0}({1}, {2})'.format(fn.__name__, pprint.pformat(args), pprint.pformat(kwargs))
            if logger:
                logger.debug(logMsg)
            elif print:
                print(logMsg)
            
            return fn(*args, **kwargs)
        
        return loggedFn
    return logArgsWrapped

# Declare types
pass

@logArgs(logger)
def GMMCluster(pts, K):
    '''
    Args:
        pts: data points to cluster
        K: number of clusters to form
    Returns:
        ClusterResult
    '''
    pass

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

    print(GMMCluster(pts, K = NClusters))
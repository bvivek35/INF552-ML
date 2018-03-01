#!/usr/bin/env python3

'''
    Class: INF552 at USC
    Submission Member: 
                        Vivek Bharadwaj <vivekb> <vivekb@usc.edu>
                        Shushyam Malige Sharanappa <maligesh> <maligesh@usc.edu>
                        Raveena Mathur <raveenam> <raveenam@usc.edu>
    This is a python example of Dimensionality Reduction using Principal Component Analysis (PCA) from the Scikit-Learn library.
'''

__version__ = '1.0'

# Imports
import os
import logging
from collections import namedtuple
from numpy import genfromtxt
from sklearn.mixture import GaussianMixture
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


PrincipalComponent = namedtuple('PrincipalComponent', ['componentNumber', 'eigVal', 'eigVec'])

@logArgsRet(logger)
def decomposeComponents(X, targetDims):
    mean = np.mean(X, axis=0)
    X_Adj = (X-mean).T
    covar = np.matmul(X_Adj, X_Adj.T) / len(X)
    eigVals, eigVecs = LA.eig(covar)
    principalComps = sorted(enumerate(eigVals), key=lambda x:x[1], reverse=True)[:targetDims]
    return [
            PrincipalComponent(count, eigVal, eigVecs[:,idx]) \
            for count, (idx, eigVal) in enumerate(principalComps)
            ]

def transform(oldPoint, components):
    tmp = np.asarray([comp.eigVec for comp in components])
    return tmp.dot(oldPoint)

if __name__ == '__main__':
    # Imports
    import sys
    import numpy as np
    import numpy.linalg as LA
    # End imports

    HELP_TEXT = 'USAGE: {0} <PCA Data File> <Target Dimension> <output file for points>'
    if len(sys.argv) != 4:
        print(HELP_TEXT.format(sys.argv[0]))
        sys.exit(1)
    else:
        inFile = sys.argv[1]
        targetDims = int(sys.argv[2])
        outFile = sys.argv[3]

    X = np.genfromtxt(inFile, delimiter='\t')

    if X.shape[1] < targetDims:
        raise ValueError('Target Dimensions {0} is greater than data dimensions {1}. Cannot Decompose!'.format(targetDims, X.shape[1]))

    principalComps = decomposeComponents(X, targetDims)
    for count, eigVal, eigVec in principalComps:
        print('Base Vector / Principal Component {0}: {1}'.format(count, eigVec))

    np.set_printoptions(suppress=True)
    with open(outFile, 'w') as fh:
        for orig in X:
            new = transform(orig, principalComps)
            fh.write('Point: {0} -> {1}'.format(orig, new))
            fh.write('\n')

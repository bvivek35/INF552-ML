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

if __name__ == '__main__':
    # Imports
    import sys
    from numpy import genfromtxt
    from sklearn.decomposition import PCA
    # End imports

    HELP_TEXT = 'USAGE: {0} <PCA Data File> <Target Dimension>'
    if len(sys.argv) != 3:
        print(HELP_TEXT.format(sys.argv[0]))
        sys.exit(1)
    else:
        inFile = sys.argv[1]
        TargetDims = int(sys.argv[2])

    X = genfromtxt(inFile, delimiter='\t')

    if X.shape[1] < TargetDims:
        raise ValueError('Target Dimensions {0} is greater than data dimensions {1}. Cannot Decompose!'.format(TargetDims, X.shape[1]))
    
    pca = PCA(n_components=TargetDims)
    pca.fit(X)
    for idx, pc in enumerate(pca.components_):
        print('Base Vector / Principal Component {0}: {1}'.format(idx, pc))
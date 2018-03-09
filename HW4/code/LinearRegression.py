import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

import sys
HELP_TEXT = 'USAGE: {0} <input file>'
if len(sys.argv) != 2:
    print(HELP_TEXT.format(sys.argv[0]))
    sys.exit(1)
else:
    inFile = sys.argv[1]

data=pd.read_csv(inFile,header=None)


def linearregression(X, y):
    return inv(X.T.dot(X)).dot(X.T).dot(y)

def normalise(X):
    n = X.shape[1]
    mean2 = np.array([np.mean(X[:,i]) for i in range(n)])
    deviation = np.array([np.std(X[:,i]) for i in range(n)])
    normalise2 = (X - mean2) / deviation

    return normalise2

y=[]

for item in data:
    y=data[2]
    x=list(zip(data[0],data[1]))
X= np.matrix(x)
#X=normalise(X)
X = np.column_stack((np.ones(len(X)), X))

#linearregression(X,y)
p=linearregression(X,y)
print('Final Weights [W0, W1, W2,...]: {0}'.format(p))

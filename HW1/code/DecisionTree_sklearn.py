#!/usr/bin/env python3

'''
    Class: INF552 at USC
    Submission Member: Vivek Bharadwaj <vivekb> <vivekb@usc.edu>

    This is a python implementation of Decision Trees using the DecisionTreeClassifier in Scikit pkg.
'''

__author__ = 'Vivek Bharadwaj'
__email__ = 'vivekb@usc.edu'
__version__ = '1.0'

import pandas as pd
from sklearn import tree

def encodeFeatures(df):
    return pd.get_dummies(df)

def encodeLabels(seriesLabels):
    '''
        Encodes the series of categorical data to integers.
        Returns the encodedSeries, decodeMap -> used to decode ints back to original labels
    '''
    uniqLabels = seriesLabels.unique()
    encodingMap = {name: idx for idx, name in enumerate(uniqLabels)}
    decodeMap = {idx: name for name, idx in encodingMap.items()}
    encodedLabels = pd.DataFrame(seriesLabels.replace(encodingMap))
    return encodedLabels, decodeMap

def encodeOneHot(df):
    '''
        Encodes the dataframe of categorical data into one-hot-encoding
        if attr has attrVal1, attrVal2, attrVal3 as unique values
        After Encoding:
        attrVal1 attrVal2 attrVal3 -> new cols
           0        1       0      -> Only 1 col will have value `1` ie. hot.

        Returns encodedFeatures, encodedLabels, decodeMap
    '''
    tmp = df[df.columns[:-1]]
    encodedFeatures = encodeFeatures(tmp)
    encodedLabels, decodeMap = encodeLabels(df[df.columns[-1]])

    return encodedFeatures, encodedLabels, decodeMap

if __name__ == '__main__':
    import sys
    import pprint

    HELP_TEXT = 'Usage: {0} <Train Data File> <Testing Data File> <Predictions Output File>'
    if len(sys.argv) != 4:
        print(HELP_TEXT.format(sys.argv[0]))
        sys.exit(1)
    else:
        trainDataFile = sys.argv[1]
        testDataFile = sys.argv[2]
        predictionsOutFile = sys.argv[3]

    # Read Training data
    df = pd.read_csv(trainDataFile, skipinitialspace=True)

    # One hot encoding for sklearn
    encodedFeatures, encodedLabels, decodeMap = encodeOneHot(df)

    # Create a DecisionTree Model and fit it to the training data
    decisionTree = tree.DecisionTreeClassifier(criterion='entropy')
    decisionTree.fit(encodedFeatures, encodedLabels)

    # Read test data
    testDF = pd.read_csv(testDataFile, skipinitialspace=True)

    # One hot encoding for sklearn
    testRecEncoded = encodeFeatures(testDF)
    # add missing cols resulted by one-hot-encoding
    for col in set(encodedFeatures.columns) - set(testRecEncoded.columns):
        testRecEncoded[col] = 0
    
    # Predict the target label
    res = decisionTree.predict(testRecEncoded)

    # Decode the label from int to the original labels
    predictions = list(enumerate(map(lambda x: decodeMap[x], res)))

    # Write prediction output
    with open(predictionsOutFile, 'w') as f:
        f.write(pprint.pformat(predictions))
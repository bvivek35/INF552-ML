#!/usr/bin/env python3

'''
    Class: INF552 at USC
    Submission Member:
                        Vivek Bharadwaj <vivekb> <vivekb@usc.edu>
                        Shushyam Malige Sharanappa <maligesh> <maligesh@usc.edu>
                        Raveena Mathur <raveenam> <raveenam@usc.edu>
    This is a python implementation of the FastMap algorithm. The distances are provided in a data file.
    The functions defined in this file require only the distances matrix to be provided.
'''

__version__ = '1.0'

# Imports
from collections import namedtuple
import copy
import numpy as np
import matplotlib.pyplot as plt
# End imports


def getFarthestPoints(distances):
    first = np.random.randint(0, 9)
    while True:
        second = np.argmax(distances[first])
        tmp = np.argmax(distances[second])
        if tmp == first:
            break
        else:
            first = second

    res = (min(first, second), max(first, second))
    return res

def getTriangularProjection(distances, line, pt):
    first, second = line
    farthest = distances[first][second]
    return (distances[first][pt]**2 + farthest**2 - distances[second][pt]**2)/(2 * farthest)

def reComputeDistances(oldDistances, lastDims):
    res = copy.deepcopy(oldDistances)
    numPoints = len(oldDistances)
    for i in range(numPoints):
        for j in range(numPoints):
            tmp = oldDistances[i][j]**2 - (lastDims[i] - lastDims[j])**2
            res[i][j] = np.sqrt(tmp)

    return res


def fastMap(distances, numDims):
    res = [[] for _ in distances]
    for _ in range(numDims):
        first, second = getFarthestPoints(distances)
        farthestDist = distances[first][second]

        for pt in range(len(distances)):
            if pt == first:
                dist = 0
            elif pt == second:
                dist = farthestDist
            else:
                dist = getTriangularProjection(distances, (first, second), pt)

            res[pt].append(dist)

        distances = reComputeDistances(distances, list(map(lambda x: x[-1], res)))

    return res

if __name__ == '__main__':
    import sys

    HELP_TEXT = '<USAGE>: {0} <distance b/w points> <number of points> <number of dims> <word list file>'
    if len(sys.argv) != 5:
        print(HELP_TEXT.format(sys.argv[0]))
        sys.exit(1)
    else:
        inFile = sys.argv[1]
        numPoints = int(sys.argv[2])
        numDims = int(sys.argv[3])
        wordFile = sys.argv[4]

    data = []
    with open(inFile) as fh:
        for line in fh:
            a, b, dist = line.split()
            data.append([int(a), int(b), float(dist)])

    distances = np.zeros((numPoints, numPoints))

    for a, b, dist in data:
        distances[a-1][b-1], distances[b-1][a-1] = dist, dist

    coOrds = fastMap(distances, numDims)

    wordList = []
    with open(wordFile) as fh:
        wordList = list(map(str.strip, fh.readlines()))

    for word, coOrd in zip(wordList, coOrds):
        print('Embed {0} - {1}'.format(word, coOrd))

    if numDims == 2:
        wordList = []
        with open(wordFile) as fh:
            wordList = list(map(str.strip, fh.readlines()))

        coOrds = np.asarray(coOrds)

        fig, ax = plt.subplots()
        ax.scatter(coOrds[:, 0], coOrds[:, 1])
        for idx, label in enumerate(wordList):
            ax.annotate(label, (coOrds[idx]))

        plt.show()
    else:
        print('Can only plot 2D !')
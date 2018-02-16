#!/usr/bin/env python3

'''
    Class: INF552 at USC
    Submission Member: Vivek Bharadwaj <vivekb> <vivekb@usc.edu>
    This is a python implementation of Decision Trees using the ID3 algorithm.

    File Structure:
    1) class DataSet # Wrapper around Pandas DataFrame. Switched to Pandas midway from using vanilla lists/dicts. So used a wrapper.
    2) class DecisionTree
        a) class Node # Base class
        b) class TreeNode:Node # Decision Node
        c) class LeafNode:Node # Leaf Node
    3) Main Program
'''

__author__ = 'Vivek Bharadwaj'
__email__ = 'vivekb@usc.edu'
__version__ = '1.0'

# Imports
import math
import pandas as pd
# End imports

class DataSet(object):
    '''
        Represents a DataSet with labels.
        Internally uses pandas dataframe.
    '''
    def __init__(self, df):
        self.df = df
        self.labelAttr = self.df.columns[-1]

    @classmethod
    def FromCSVFile(cls, filePath):
        return cls(pd.read_csv(filePath, skipinitialspace=True))

    def partitionBy(self, attr, attrVal, negate=False):
        res = None
        if negate:
            res = self.df.loc[self.df[attr] != attrVal]
        else:
            res = self.df.loc[self.df[attr] == attrVal]
        return DataSet(res)
    
    def removeAttr(self, attr):
        return DataSet(self.df.drop(attr, axis=1))
    
    def isPure(self):
        # return len(self.df[self.labelAttr].unique()) == 1
        return len(self.df.columns) == 1 or len(self.df[self.labelAttr].unique()) == 1
    
    def uniqueAttrVals(self, attr):
        return self.df[attr].unique()
    
    def getLabelPair(self):
        print('LabelPair: \n', self.df)
        if self.df.empty:
            return (self.labelAttr, 'undefined')
        return (self.labelAttr, self.df[self.labelAttr].iloc[0])
    
    def getAttrs(self):
        return self.df.columns[:-1]
    
    def computeEntropy(self):
        '''
            Rough Algorithm to compute entropy
            ##########
            Computing entropy wrt labelAttr.
            Let labelAttr has [labelAttrVal1, labelAttrVal2, ...]
            denominator = len(self.df)
            totalEntropy = 0
            for labelAttrVal in labelAttr.Values:
                tmp = len(self.partitionBy(labelAttr, labelAttrVal))
                frac = tmp/denominator
                entropy = - frac * math.log(frac, 2)
                totalEntropy += entropy
            ########
        '''
        print('Entropy for: \n', self.df)
        denominator = len(self.df)
        totalEntropy = 0
        for labelAttrVal in self.uniqueAttrVals(self.labelAttr):
            tmp = len(self.partitionBy(self.labelAttr, labelAttrVal))
            frac = tmp/denominator
            entropy = - frac * math.log(frac, 2)
            totalEntropy += entropy
        
        return totalEntropy
    
    def iter(self):
        return self.df.iterrows()
    
    @staticmethod
    def __parseCSVStr(s):
        return list(
                    map(lambda r: list(
                                    map(str.strip, 
                                        r.strip().split(','))
                                    ), 
                        s.strip().split('\n'))
                    )

    def __len__(self):
        return len(self.df)

    def __str__(self):
        return repr(self)
    
    def __repr__(self):
        return repr(self.__dict__)

class DecisionTree(object):
    '''
        Represents a Decision Tree which can be trained and used for classification.
    '''
    class Node(object):
        '''
            Represents a generic node of a tree.
        '''
        def __init__(self, attr):
            self.attr = attr

        def isLeaf(self):
            return False

        def __str__(self):
            return repr(self)

        def __repr__(self):
            return repr(self.__dict__)
        
    class DecisionNode(Node):
        '''
            Represents a decision node in the tree ie. not a leaf.
            children is a list of child Nodes
            childIdx is a map of attrVal : idxInChildrenList
        '''
        def __init__(self, attr, children, childIdx):
            super().__init__(attr)
            self.children = children
            self.childIdx = childIdx

        def getChild(self, attrVal):
            '''
                Gets the child for the node whose edge from this node to the child follows attrVal
            '''
            return self.children[self.childIdx(attrVal)]
        
        def addChild(self, attrVal, child):
            self.children.append(child)
            self.childIdx[attrVal] = len(self.children)-1

        def __str__(self):
            return repr(self)

        def __repr__(self):
            return repr(self.__dict__)

    class LeafNode(Node):
        '''
            Represents the leaf node in the tree.
        '''
        def __init__(self, attr, attrVal):
            super().__init__(attr)
            self.attrVal = attrVal
        
        def isLeaf(self):
            return True

        def __str__(self):
            return repr(self)

        def __repr__(self):
            return repr(self.__dict__)


    def __init__(self, root):
        self.root = root
        

    @classmethod
    def buildTreeFromDataSet(cls, dataSet):
        root = DecisionTree.__buildTreeFromDataSet_R(dataSet)
        return cls(root)

    @classmethod
    def deserializeTree(cls, serialized):
        pass

    @staticmethod
    def mkLeafNode(attr, attrVal):
        return DecisionTree.LeafNode(attr, attrVal)
    
    @staticmethod
    def mkDecisionNode(attr):
        return DecisionTree.DecisionNode(attr, [], {})

    @staticmethod
    def __buildTreeFromDataSet_R(dataSet):
        '''
            Builds the decision tree recursively using the ID3 algorithm.
            Returns the root of the tree.
        '''
        root = None
        if dataSet.isPure():
            root = DecisionTree.mkLeafNode(*dataSet.getLabelPair())
        else:
            attr = DecisionTree.__getBestAttr(dataSet)
            print('Chose attr: ', attr)
            root = DecisionTree.mkDecisionNode(attr)
            for attrVal in dataSet.uniqueAttrVals(attr):
                tmp = dataSet.partitionBy(attr, attrVal).removeAttr(attr)
                print('Trying building with: ', attrVal, '\n', tmp.df)
                root.addChild(attrVal, 
                    DecisionTree.__buildTreeFromDataSet_R(tmp))
        
        return root
    
    @staticmethod
    def __getBestAttr(dataSet):
        return DecisionTree.__getBestAttr_InformationGainStrategy(dataSet)
    
    # RandomStrategy is not used !!!!
    @staticmethod
    def __getBestAttr_RandomStrategy(dataSet):
        import random
        tmp = len(dataSet.df.columns)
        print('Choosing among: \n', dataSet.df.columns)
        idx = random.randint(0, tmp-2)
        return dataSet.df.columns[idx]
    
    @staticmethod
    def __getBestAttr_InformationGainStrategy(dataSet):
        '''
            InformationGainBefore = dataSet.computeEntropy()
            best_attr = max(InfomationGainAfter(attr) - InformationGainBefore(attr) for each attr in dataSet)

            Rough Algorithm
            #########
            iGainBefore = dataSet.computeEntropy()
            maxIGain, bestAttr = (float("-inf"), None)
            for attr in dataSet.getAttrs():
                iGainAfter = 0
                for attrVal in dataSet.uniqueAttrVals(attr):
                    tmp = dataSet.partitionBy(attr, attrVal, negate=False)
                    iGainAfter += tmp.computeEntropy()
                
                iGain = iGainAfter - iGainBefore
                if maxIGain < iGain:
                    maxIGain, bestAttr = iGain, attr
            
            return bestAttr
            #########
        '''
        iGainBefore = dataSet.computeEntropy()
        wtDenom = len(dataSet)
        print('iGainBefore : ', iGainBefore)
        maxIGain, bestAttr = (float("-inf"), None)
        for attr in dataSet.getAttrs():
            iGainAfter = 0
            for attrVal in dataSet.uniqueAttrVals(attr):
                tmp = dataSet.partitionBy(attr, attrVal, negate=False)
                wt = len(tmp) / wtDenom
                iGainAfter += wt * tmp.computeEntropy()
            print('IGain for : ', attr, ' : ', iGainAfter)
            
            iGain = iGainBefore - iGainAfter
            if maxIGain < iGain: # This ensures that if there is a tie b/w 2 attrs, the first one wins. 
                maxIGain, bestAttr = iGain, attr
        
        return bestAttr

    def predict(self, dataSet):
        '''
            dataSet here does NOTT have the LAST COLUMN (labelAttr)
        '''
        res = []
        for index, row in dataSet.iter():
            res.append(self.traverseToLeaf(row))
        return res
    
    def traverseToLeaf(self, row):
        '''
            Traverses the tree and returns the last leaf node. 
            If not such node is found, returns 'No Prediction'
        '''
        root = self.root
        while not root.isLeaf():
            attr = root.attr
            val = row[attr]
            print('Following attr: ', attr, ' : ', val)
            print(root.childIdx)
            if val in root.childIdx: # if this node has a child corresponding to the attrVal of argument
                idx = root.childIdx[val]
                root = root.children[idx]
            else:
                print('No Prediction found for \n', row)
                return 'No Prediction!'
        print(root)
        return root.attrVal
    
    def toGraphVizString(self):
        s = ''
        return s

    def stringify(self):
        return self.__stringify_R(self.root)
    
    def __stringify_R(self, root):
        '''
            Prints the tree in DFS style.
        '''
        s = ''
        if root:
            Q = [(0, '', root)]
            while Q:
                lvl, attrVal, node = Q.pop()
                print('Trying ', lvl, attrVal, node.attr)
                s += lvl * '\t'
                if attrVal:
                    s += attrVal + ':'
                s += node.attr
                if node.isLeaf():
                    s += ':' + node.attrVal
                else:
                    for nodeVal in node.childIdx:
                        Q.append((lvl+1, nodeVal, node.children[node.childIdx[nodeVal]]))
                
                s += '\n'
        return s

    def __str__(self):
        return repr(self)
    
    def __repr__(self):
        return repr(self.__dict__)


if __name__ == '__main__':
    import sys
    import pprint

    HELP_TEXT = 'Usage: {0} <Train Data File> <Testing Data File> <Predictions Output File> <Tree Output File>'
    if len(sys.argv) != 5:
        print(HELP_TEXT.format(sys.argv[0]))
        sys.exit(1)
    else:
        trainDataFile = sys.argv[1]
        testDataFile = sys.argv[2]
        predictionsOutFile = sys.argv[3]
        treeOutFile = sys.argv[4]

    # Monkey Patching print to disable any debug prints
    def noOpFn(*args, **kwargs):
        pass    
    print_orig = print
    print = noOpFn
    # END Monkey Patching

    # Read training data
    dataSet = DataSet.FromCSVFile(trainDataFile)
    
    # Build the decision tree using training data
    decisionTree = DecisionTree.buildTreeFromDataSet(dataSet)

    # Write the string representation of the tree to file
    treeString = decisionTree.stringify()
    with open(treeOutFile, 'w') as outFile:
        outFile.write(treeString)

    # Read test data
    testDataSet = DataSet.FromCSVFile(testDataFile)
    
    # Make predictions on the test data
    predictions = decisionTree.predict(testDataSet)

    # Write predictions to file
    with open(predictionsOutFile, 'w') as predictFile:
        predictFile.write(pprint.pformat(list(enumerate(predictions))))
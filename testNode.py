# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 2014
@author: teddy
"""
import numpy as np

rng = np.random
from Node import *


def main():
    myNode = Node(1, [2, 2])  # LayerNum=1, LayerPos=[2,2]
    # Prepare AlgParams,InitNodeBelief,InitNodeLearnedFeatures
    N = 400
    feats = 784
    AlgParams = {}
    D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
    AlgParams['D'] = D  # an initial random input
    AlgParams['N'] = 400
    AlgParams['feats'] = feats
    AlgParams['training_steps'] = 10000
    AlgorithmChoice = 'LogRegression'
    AlgParams['w'] = theano.shared(rng.randn(feats), name="w")
    myNode.loadInput(D)
    myNode.initNodeLearningParams(AlgorithmChoice, AlgParams)
    # myNode.doNodeLearning('training')
    #myLearningAlgorithm = LearningAlgorithm(AlgParams)
    print "Multi-dim Nodes example"
    Row = 4
    Col = 4
    LayerNum = 2
    NodeArray = [[Node(LayerNum, [i, j]) for j in range(Row)] for i in range(Col)]
    for I in range(Row):
        for J in range(Col):
            print(NodeArray[I][J].NodePosition)
            NodeArray[I][J].loadInput(D)
            myNode.initNodeLearningParams(AlgorithmChoice, AlgParams)
            #myNode.doNodeLearning('training')

    print(type(myNode.Belief))


main()

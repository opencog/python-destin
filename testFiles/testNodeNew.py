# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 2014
@author: teddy
"""

import numpy as np

import theano
import theano.tensor as T


rng = np.random


def LogReg(D, N, training_steps, feats, w):
    # Declare Theano symbolic variables
    x = T.matrix("x")
    y = T.vector("y")
    b = theano.shared(0., name="b")
    print "Initial model:"
    print w.get_value(), b.get_value()

    # Construct Theano expression graph
    p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))  # Probability that target = 1
    prediction = p_1 > 0.5  # The prediction thresholded
    xent = -y * T.log(p_1) - (1 - y) * T.log(1 - p_1)  # Cross-entropy loss function
    cost = xent.mean() + 0.01 * (w ** 2).sum()  # The cost to minimize
    gw, gb = T.grad(cost, [w, b])  # Compute the gradient of the cost
    # (we shall return to this in a
    # following section of this tutorial)

    # Compile
    train = theano.function(
        inputs=[x, y],
        outputs=[prediction, xent],
        updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
    predict = theano.function(inputs=[x], outputs=prediction)

    # Train
    for i in range(training_steps):
        pred, err = train(D[0], D[1])

        # print "Final model:"
        # print w.get_value(), b.get_value()
        #print "target values for D:", D[1]
        #print "prediction on D:", predict(D[0])
        #return [w*D[0],w] # returns Learned Weigths and Belief W*Input


class LearningAlgorithm:
    def __init__(self, AlgParams):
        self.D = AlgParams['D']
        self.N = AlgParams['N']
        self.training_steps = AlgParams['training_steps']
        self.feats = AlgParams['feats']
        self.w = AlgParams['w']

    def runLearningAlgorithm(self, Mode):
        LogReg(self.D, self.N, self.training_steps, self.feats, self.w)


class Node:
    def __init__(self, LayerNumber, NodePos):
        self.LayerNumber = LayerNumber
        self.NodePosition = NodePos

    def initLearningAlgorithm(self, AlgorithmChoice, AlgParams, InitNodeBelief, InitNodeLearnedFeatures):
        self.AlgorithmChoice = AlgorithmChoice
        if AlgorithmChoice == 'LogRegression':
            self.AlgorithmChoice = AlgorithmChoice  # Name of the Algorithm
            self.Belief = InitNodeBelief
            self.LearnedFeatures = InitNodeLearnedFeatures
            # Attrbutes For the learning Algorithm Class
            self.LearningAlgorithm = LearningAlgorithm(AlgParams)
            self.LearningAlgorithm.D = AlgParams['D']
            self.LearningAlgorithm.N = AlgParams['N']
            self.LearningAlgorithm.training_steps = AlgParams['training_steps']
            self.LearningAlgorithm.feats = AlgParams['feats']
            self.LearningAlgorithm.w = AlgParams['w']
        else:
            print('make sure that you are choosing an available learning algorithm')
            print('python is exitting')
            exit(0)

    def loadInput(self, Input):
        self.Input = Input

    def doLearning(self, Mode):
        self.LearningAlgorithm.runLearningAlgorithm(Mode)
        self.Belief = self.LearningAlgorithm.w * self.LearningAlgorithm.D[0]
        self.LearnedFeatures = self.LearningAlgorithm.w

        # Mode Differentiates Training from Testing
        # Mode == 1 Training: there will be update of Model/ or parameters
        # Mode == 0 Testing: No model-updating only encoding


def main():
    # Declare Node
    myNode = Node(1, [2, 2])  # LayerNum=1, LayerPos=[2,2]
    # Prepare AlgParams,InitNodeBelief,InitNodeLearnedFeatures
    N = 400
    feats = 784
    AlgParams = {}
    D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
    AlgParams['D'] = D
    AlgParams['N'] = 400
    AlgParams['feats'] = feats
    AlgParams['training_steps'] = 10000
    AlgParams['w'] = theano.shared(rng.randn(feats), name="w")
    myLearningAlgorithm = LearningAlgorithm(AlgParams)
    InitNodeBelief = AlgParams['w'] * D[0]
    InitNodeLearnedFeatures = AlgParams['w']
    myNode.initLearningAlgorithm('LogRegression', AlgParams, InitNodeBelief, InitNodeLearnedFeatures)
    myNode.loadInput(D)
    myNode.doLearning(1)


main()

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 2014
@author: teddy
"""
import numpy as np
import scipy.io as io
from random import randrange
import theano
from theano import function
import theano.tensor as T

rng = np.random
from LearningAlgorithm import *
from Node import *
from Layer import *


def main():
    myLayer = Layer(0, [8, 8], 'Adjacent', ImageType='Gray')
    N = 1
    feats = 16
    img = np.random.rand(32, 32)
    Label = 1
    Ratio = 4
    myLayer.loadInput(img, Ratio)
    # Initialize LayerLearningAlgorithm by specifying the ff
    # AlgorithmChoice,AlgParams,InitNodeBelief,InitNodeLearnedFeatures
    AlgorithmChoice = 'LogRegression'
    AlgParams = {}
    # AlgParams['N'] =
    D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
    AlgParams['D'] = D
    AlgParams['N'] = N
    AlgParams['feats'] = feats
    AlgParams['training_steps'] = 1
    w = theano.shared(rng.randn(feats), name="w")
    AlgParams['w'] = w
    InitNodeLearnedFeatures = w
    InitNodeBelief = w * img
    myLayer.initLayerLearningParams(AlgorithmChoice, AlgParams)


main()

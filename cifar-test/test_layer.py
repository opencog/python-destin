# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 2014
@author: teddy
"""
import theano

rng = np.random
from layer import *


def main():
    myLayer = Layer(0, [8, 8], 'Adjacent', image_type='Gray')
    N = 1
    feats = 16
    img = np.random.rand(32, 32)
    Label = 1
    Ratio = 4
    myLayer.load_input(img, Ratio)
    # Initialize Layerlearning_algorithm by specifying the ff
    # algorithm_choice,alg_params,InitNodebelief,InitNodeLearnedFeatures
    algorithm_choice = 'LogRegression'
    alg_params = {}
    # alg_params['N'] =
    D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
    alg_params['D'] = D
    alg_params['N'] = N
    alg_params['feats'] = feats
    alg_params['training_steps'] = 1
    w = theano.shared(rng.randn(feats), name="w")
    alg_params['w'] = w
    InitNodeLearnedFeatures = w
    InitNodebelief = w * img
    myLayer.init_layer_learning_params(algorithm_choice, alg_params)


main()

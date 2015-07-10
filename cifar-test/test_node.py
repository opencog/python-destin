# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 2014
@author: teddy
"""
import sys
sys.path.append('../destin/')
import numpy as np

rng = np.random
from node import *



def main():
    myNode = Node(1, [2, 2])  # layer_num=1, LayerPos=[2,2]
    # Prepare alg_params,InitNodebelief,InitNodeLearnedFeatures
    N = 400
    feats = 784
    alg_params = {}
    D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
    alg_params['D'] = D  # an initial random input
    alg_params['N'] = 400
    alg_params['feats'] = feats
    alg_params['training_steps'] = 10000
    algorithm_choice = 'LogRegression'
    alg_params['w'] = theano.shared(rng.randn(feats), name="w")
    myNode.load_input(D)
    myNode.init_node_learning_params(algorithm_choice, alg_params)
    # myNode.do_node_learning('training')
    #mylearning_algorithm = learning_algorithm(alg_params)
    print "Multi-dim nodes example"
    Row = 4
    Col = 4
    layer_num = 2
    NodeArray = [[Node(layer_num, [i, j]) for j in range(Row)]
                 for i in range(Col)]
    for I in range(Row):
        for J in range(Col):
            print(NodeArray[I][J].node_position)
            NodeArray[I][J].load_input(D)
            myNode.init_node_learning_params(algorithm_choice, alg_params)
            # myNode.do_node_learning('training')

    print(type(myNode.belief))


main()

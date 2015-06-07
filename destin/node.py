# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 2014
@author: teddy

@note: An implementation of node of the DeSTIN architecture
"""

from clustering import *
from auto_encoder import *


class Node:
    """
    This class describes all nodes of the DeSTIN network
    """
    def __init__(self, 
                 layer_number, 
                 node_pos, 
                 cifar_stat):
        """
        Initialize each node 

        @param layer_number: layer number  
        @param node_pos: position of nodes 
        @param cifar_stat: parameters needed for the CIFAR dataset
        """
        self.layer_number = layer_number
        self.node_position = node_pos
        self.belief = []
        # cifarStat = load_cifar(4)#  to be used for Normalization and Whitening
        #  Purposes
        self.patch_mean = cifar_stat['patch_mean']
        self.patch_std = cifar_stat['patch_std']
        self.v = cifar_stat['whiten_mat']

    def init_node_learning_params(self, algorithm_choice, alg_params):
        """
        Initialize learning parameters for all nodes 

        @param algorithm_choice: choice of algorithm 
        @param alg_params: Dictionary of various parameters needed for learning algorithm 
        """
        self.algorithm_choice = algorithm_choice
        if algorithm_choice == 'Clustering':
            cents_per_layer = alg_params['num_cents_per_layer']
            #  input_width = input_widths[layer_num]
            if self.layer_number == 0:
                input_width = 48
            else:
                input_width = cents_per_layer[self.layer_number - 1] * 4
            self.learning_algorithm = Clustering(mr=alg_params['mr'], 
                                                 vr=alg_params['vr'], 
                                                 sr=alg_params['sr'], 
                                                 di=input_width,
                                                 ce=alg_params['num_cents_per_layer'][self.layer_number], 
                                                 node_id=self.node_position)
        else:
            self.belief = np.ones((alg_params[self.layer_number][1], 1))
            self.algorithm_choice = algorithm_choice
            self.learning_algorithm = NNSAE(
                alg_params[self.layer_number][0], alg_params[self.layer_number][1])

    def load_input(self, In):
        """
        Load all input data - image patches, and normalize them 

        @param In: input data 
        """
        if self.layer_number == 0:
            In = In - self.patch_mean
            In = In / self.patch_std
            In = In.dot(self.v)
        self.input = In

    def do_node_learning(self, mode):
        """
        Peform learning on each node 

        @param mode: True indicates that the network is training
        """
        if self.algorithm_choice == 'Clustering':
            self.learning_algorithm.update_node(self.input, mode)
            self.belief = self.learning_algorithm.belief
        else:
            self.learning_algorithm.train(self.input)
            W = np.transpose((self.learning_algorithm.W + 0.00005)/np.sum((self.learning_algorithm.W + 0.00005),0))
            input_ = self.input/np.sum(self.input,0)
            # input_ = np.transpose(self.input)/np.sum(np.transpose(self.input),0)
            # print np.shape(W)
            # print np.shape(input_)
            # activations = np.dot(W, input_) + 0.00005
            dist = W - input_
            sq_dist = np.square(dist)
            norm_dist = np.sum(sq_dist, axis=1)
            chk = (norm_dist == 0)
            if any(chk):
                self.belief = np.zeros_like(self.belief)
                self.belief[chk] = 1.0
            else:
                norm_dist = 1 / norm_dist
                belief = (norm_dist / sum(norm_dist)) # .reshape(np.shape(self.belief)[0], np.shape(self.belief)[1])
            #belief = activations / (np.sum(activations))
            self.belief = belief
            # belief = np.maximum(activations, 0)
            # self.belief = belief
            # for K in range(activations.shape[0]):
            #     belief[K] = max(0.0, float((activations[K] - 0.025)))
            # self.belief = np.asarray(belief)/np.sum(belief)

            """

    def calcuatebelief(self, input_):
        self.load_inputTonodes(input_, [4,4])
        for I in range(len(self.nodes)):
            for J in range(len(self.nodes[0])):
                W = np.transpose(self.NNSAE.W)
                Image = return_node_input(input_, [I*4, J*4], [4,4], 'Adjacent', 'Color')
                activations = np.dot(W, np.transpose(Image))
                activations = activations/sum(sum(activations))
                self.nodes[I][J].Activation = activations
                m = np.mean(np.mean(activations,1))
                for K in range(activations.shape[0]):
                        self.nodes[I][J].belief[K,0] = max(0, (activations[K,0] - 0.025))
                print self.nodes[0][0].belief"""
            """
                def produce_belief(self, sqdiff):
        """
        # Update belief state.
        """
        normdist = np.sum(sqdiff / self.var, axis=1)
        chk = (normdist == 0)
        if any(chk):
            self.belief = np.zeros((1, self.CENTS))
            self.belief[chk] = 1.0
        else:
            normdist = 1 / normdist
            self.belief = (normdist / sum(normdist)).reshape(1, self.CENTS)

            """

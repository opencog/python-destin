# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 2014
@author: teddy
"""

from Clustering import *
from AutoEncoder import *


class Node:

    def __init__(self, layer_number, node_pos, cifar_stat={'patch_mean': [], 'patch_std': [], 'whiten_mat': []}):
        self.layer_number = layer_number
        self.node_position = node_pos
        self.belief = []
        # cifarStat = load_cifar(4)#  to be used for Normalization and Whitening
        #  Purposes
        self.patch_mean = cifar_stat['patch_mean']
        self.patch_std = cifar_stat['patch_std']
        self.v = cifar_stat['whiten_mat']

    def init_node_learning_params(self, algorithm_choice, alg_params):
        self.algorithm_choice = algorithm_choice
        if algorithm_choice == 'Clustering':
            cents_per_layer = alg_params['num_cents_per_layer']
            #  input_width = input_widths[layer_num]
            if self.layer_number == 0:
                input_width = 48
            else:
                input_width = cents_per_layer[self.layer_number - 1] * 4
            self.learning_algorithm = Clustering(alg_params['mr'], alg_params['vr'], alg_params['sr'], input_width,
                                                alg_params['num_cents_per_layer'][self.layer_number], self.node_position)
            # mr, vr, sr, di, ce, node_id
        else:
            self.belief = np.ones((alg_params[self.layer_number][1], 1))
            self.algorithm_choice = algorithm_choice
            self.learning_algorithm = NNSAE(
                alg_params[self.layer_number][0], alg_params[self.layer_number][1])

    def load_input(self, In):
        if self.layer_number == 0:
            In = In - self.patch_mean
            In = In / self.patch_std
            In = In.dot(self.v)
        self.input = In

    def do_node_learning(self, mode):
        if self.algorithm_choice == 'Clustering':
            self.learning_algorithm.update_node(self.input, mode)
            self.belief = self.learning_algorithm.belief
        else:
            self.learning_algorithm.train(self.input)
            W = np.transpose((self.learning_algorithm.W + 0.00005)/np.sum((self.learning_algorithm.W + 0.00005),0))
            input = np.transpose(self.input)/np.sum(np.transpose(self.input),0)
            Activations = np.dot(W, input) + 0.00005
            belief = Activations / (np.sum(Activations))
            self.belief = belief
            # belief = np.maximum(Activations, 0)
            # self.belief = belief
            # for K in range(Activations.shape[0]):
            #     belief[K] = max(0.0, float((Activations[K] - 0.025)))
            # self.belief = np.asarray(belief)/np.sum(belief)

            """

    def calcuatebelief(self, input):
        self.load_inputTonodes(input, [4,4])
        for I in range(len(self.nodes)):
            for J in range(len(self.nodes[0])):
                W = np.transpose(self.NNSAE.W)
                Image = return_node_input(input, [I*4, J*4], [4,4], 'Adjacent', 'Color')
                Activations = np.dot(W, np.transpose(Image))
                Activations = Activations/sum(sum(Activations))
                self.nodes[I][J].Activation = Activations
                m = np.mean(np.mean(Activations,1))
                for K in range(Activations.shape[0]):
                        self.nodes[I][J].belief[K,0] = max(0, (Activations[K,0] - 0.025))
                print self.nodes[0][0].belief"""

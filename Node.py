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
        self.patch_mean = cifar_stat['patch_mean']
        self.patch_std = cifar_stat['patch_std']
        self.v = cifar_stat['whiten_mat']
        self.learning_algorithm = []
        self.input = []
        self.algorithm_choice = []

    def init_node_learning_params(self, algorithm_choice, alg_params):
        self.algorithm_choice = algorithm_choice
        if algorithm_choice == 'Clustering':
            cents_per_layer = alg_params['num_cents_per_layer']

            if self.layer_number == 0:
                input_width = 48
            else:
                input_width = cents_per_layer[self.layer_number - 1] * 4
            self.learning_algorithm = Clustering(alg_params['mr'], alg_params['vr'], alg_params['sr'], input_width,
                                                alg_params['num_cents_per_layer'][self.layer_number], self.node_position)
        else:
            self.belief = np.ones((alg_params[self.layer_number][1], 1))
            self.algorithm_choice = algorithm_choice
            self.learning_algorithm = NNSAE(
                alg_params[self.layer_number][0], alg_params[self.layer_number][1])

    def load_input(self, in_):
        if self.layer_number == 0:
            in_ = in_ - self.patch_mean
            in_ = in_ / self.patch_std
            in_ = in_.dot(self.v)
        self.input = in_

    def do_node_learning(self, mode):
        if self.algorithm_choice == 'Clustering':
            self.learning_algorithm.update_node(self.input, mode)
            self.belief = self.learning_algorithm.belief
        else:
            self.learning_algorithm.train(self.input)
            eps = np.exp(-10)
            weight = np.transpose((self.learning_algorithm.W + eps)/np.sum((self.learning_algorithm.W + eps), 0))
            input_ = self.input/np.sum(self.input, 0)
            dist = weight - input_
            sq_dist = np.square(dist)
            norm_dist = np.sum(sq_dist, axis=1)
            chk = (norm_dist == 0)
            if any(chk):
                self.belief = np.zeros_like(self.belief)
                self.belief[chk] = 1.0
            else:
                norm_dist = 1 / norm_dist
                belief = (norm_dist / sum(norm_dist))
            self.belief = belief/sum(belief)    # Make sure that beliefs are normalized

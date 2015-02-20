# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 2014
@author: teddy
"""
from node import *
from load_data import *
class Layer:
    def __init__(self, layer_num, number_of_nodes, cifar_stat, patch_mode='Adjacent', image_type='Color'):
        self.patch_mode = patch_mode
        self.image_type = image_type
        self.layer_number = layer_num
        self.number_of_nodes = number_of_nodes  # Usually a list with two elements
        Row = number_of_nodes[0]
        Col = number_of_nodes[1]
        if layer_num == 0:
            nodes = [[Node(layer_num, [i, j], cifar_stat) for j in range(Row)] for i in range(Col)]
        else:
            nodes = [[Node(layer_num, [i, j], cifar_stat) for j in range(Row)] for i in range(Col)]
        self.nodes = nodes


    def load_input(self, input_, R):
        Ratio = R[0]
        # Ratio equals to the number of lower layer units getting combined and being fed to the upper layer
        if self.layer_number == 0:
            Nx = 0  # X coordinate of the current node
            for I in range(0, input_.shape[0], Ratio):
                Ny = 0  # Y coordinate of the current node
                for J in range(0, input_.shape[1], Ratio):
                    self.nodes[Nx][Ny].load_input(return_node_input(input_, [I, J], Ratio, self.patch_mode,
                                                                 self.image_type))  # returns inputs to the node located at Position [Nx,Ny]
                    Ny += 1
                Nx += 1
        else:
            Nx = 0  # X coordinate of the current node
            Ny = 0  # Y coordinate of the current node
            for I in range(0, len(input_[0]), Ratio):
                Ny = 0
                for J in range(0, len(input_[1]), Ratio):
                    input_temp = np.array([])
                    for K in range(I, I + Ratio):
                        for L in range(J, J + Ratio):
                            input_temp = np.append(input_temp, np.asarray(input_[K][
                                                                            L].belief))# Combine the beliefs of the nodes passed
                    self.nodes[Nx][Ny].load_input(np.ravel(input_temp))
                    Ny += 1
                Nx += 1

    def init_layer_learning_params(self, algorithm_choice, alg_params):
        for I in range(len(self.nodes)):
            for J in range(len(self.nodes[0])):
                self.nodes[I][J].init_node_learning_params(algorithm_choice, alg_params)

    def do_layer_learning(self):
        for I in range(len(self.nodes)):
            for J in range(len(self.nodes[0])):
                self.nodes[I][J].do_node_learning(self.mode)

    def train_typical_node(self, input_, windowSize, algorithm_choice):
        TN = self.nodes[0][0]
        [H, V] = windowSize
        if self.layer_number == 0:
            X = input_.shape[0] - H + 1
            Y = input_.shape[1] - V + 1
            for I in range(X):
                for J in range(Y):
                    TN.load_input(
                        return_node_input(input_, [I, J], H, self.patch_mode, self.image_type))
                    TN.do_node_learning(self.mode)
        else:
            X = len(input_[0]) - H + 1
            Y = len(input_[1]) - V + 1
            for I in range(X):
                for J in range(Y):
                    input_temp = np.array([])
                    for K in range(I, I + H):
                        for L in range(J, J + V):
                            input_temp = np.append(
                                input_temp, np.array(np.ravel(input_[K][L].belief)))
                            # Combine the beliefs of the nodes passed
                    TN.load_input(np.ravel(input_temp))
                    TN.do_node_learning(self.mode)
        self.nodes[0][0] = TN

    def share_learned_parameters(self):
        for I in range(len(self.nodes)):
            for J in range(len(self.nodes[0])):
                self.nodes[I][J] = self.nodes[0][0]

    def update_beliefs(self):
        for I in range(len(self.nodes)):
            for J in range(len(self.nodes[0])):
                self.nodes[I][J].do_node_learning(False)

# -*- coding: utf-8 -*-
__author__ = 'teddy'
import scipy.io as io
from load_data import *
from layer import *

# io.savemat(file_name,Dict,True)
# TODO: get ridoff the sequential requirements like first feed the layer
# an input the you can initialize it


class Network():

    def __init__(self, num_layers, alg_choice, alg_params, num_nodes_per_layer, cifar_stat, patch_mode='Adjacent', image_type='Color'):
        self.network_belief = {}
        self.lowest_layer = 1
        # this is going to store beliefs for every image DeSTIN sees
        self.network_belief['belief'] = np.array([])
        self.save_belief_option = 'True'
        self.belief_file_name = 'beliefs.mat'
        self.number_of_layers = num_layers
        self.algorithm_choice = alg_choice
        self.algorithm_params = alg_params
        self.number_of_nodesPerLayer = num_nodes_per_layer
        self.patch_mode = patch_mode
        self.image_type = image_type
        self.layers = [
            [Layer(j, num_nodes_per_layer[j], cifar_stat, self.patch_mode, self.image_type) for j in range(num_layers)]]

    def setmode(self, mode):
        self.operating_mode = mode
        for I in range(self.number_of_layers):
            self.layers[0][I].mode = mode

    def init_network(self):
        for L in range(self.number_of_layers):
            self.initLayer(L)

    def set_lowest_layer(self, lowest_layer):
        self.lowest_layer = lowest_layer

    def initLayer(self, layer_num):
        self.layers[0][layer_num].init_layer_learning_params(
            self.algorithm_choice, self.algorithm_params)

    def train_layer(self, layer_num):
        self.layers[0][layer_num].do_layer_learning(self.operating_mode)

    def update_belief_exporter(self):
        for i in range(self.lowest_layer, self.number_of_layers):
            for j in range(len(self.layers[0][i].nodes)):
                for k in range(len(self.layers[0][i].nodes[0])):
                    if self.network_belief['belief'] == np.array([]):
                        self.network_belief['belief'] = np.array(
                            self.layers[0][i].nodes[j][k].belief).ravel()
                    else:
                        self.network_belief['belief'] = np.hstack((np.array(self.network_belief['belief']),
                                                                  np.array(self.layers[0][i].nodes[j][k].belief).ravel()))

    def dump_belief(self, num_of_images):
        total_belief_len = len(np.array(self.network_belief).ravel())
        single_belief_len = total_belief_len / num_of_images
        print np.array(self.network_belief).ravel()
        belief = np.array(self.network_belief).reshape(
            num_of_images, single_belief_len)
        io.savemat(self.belief_file_name, belief)

    def clean_belief_exporter(self):
        self.network_belief['belief'] = np.array([])

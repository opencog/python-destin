# -*- coding: utf-8 -*-
__author__ = 'teddy'
"""
@note: An implementation of DeSTIN Network architecture 
"""
import scipy.io as io
from load_data import *
from layer import *

# io.savemat(file_name,Dict,True)
# TODO: get ridoff the sequential requirements like first feed the layer
# an input the you can initialize it


class Network():
    """
    An implementation of DeSTIN 

    This class creates the network. 
    """

    def __init__(self, 
                 num_layers, 
                 alg_choice, 
                 alg_params, 
                 num_nodes_per_layer, 
                 cifar_stat, 
                 patch_mode='Adjacent', 
                 image_type='Color'):

        """
        Initialize DeSTIN

        @param num_layers: number of layers 
        @param alg_choice: choice of algorithm - one from ["Clustering", "Auto-Encoder", "LogRegresion"]
        @param alg_params: Dictionary of various parameters needed for learning algorithm 
        @param num_nodes_per_layer: number of nodes present in each layer of DeSTIN
        @param cifar_stat: parameters needed for the CIFAR dataset
        @param patch_mode: patch selection mode for images 
        @param image_type: type of images to be used
        """

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
        """
        Sets the network training mode

        @param mode: True indicates that the network is training
        """
        self.operating_mode = mode
        for I in range(self.number_of_layers):
            self.layers[0][I].mode = mode

    def init_network(self):
        """
        Initialize all layers of the network
        """
        for L in range(self.number_of_layers):
            self.initLayer(L)

    def set_lowest_layer(self, lowest_layer):
        """
        Sets the indicated layer to be the lowest one

        @param lowest_layer: value of lowest network layer i.e. starting layer
        """
        self.lowest_layer = lowest_layer

    def initLayer(self, layer_num):
        """
        Initialize all layers with the learning parameters

        @param layer_num: layer number 
        """
        self.layers[0][layer_num].init_layer_learning_params(
            self.algorithm_choice, self.algorithm_params)

    def train_layer(self, layer_num):
        """
        Train all layers based on the network mode

        @param layer_num: layer number to train 
        """
        self.layers[0][layer_num].do_layer_learning(self.operating_mode)

    def update_belief_exporter(self):
        """
        Updates the belief states across the network
        """
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
        """
        Saving belief states for all images in a file 

        @param num_of_images: number of images 
        """
        total_belief_len = len(np.array(self.network_belief).ravel())
        single_belief_len = total_belief_len / num_of_images
        print np.array(self.network_belief).ravel()
        belief = np.array(self.network_belief).reshape(
            num_of_images, single_belief_len)
        io.savemat(self.belief_file_name, belief)

    def clean_belief_exporter(self):
        """
        Delete all beliefs i.e. clean instantiation
        """
        self.network_belief['belief'] = np.array([])

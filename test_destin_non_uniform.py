__author__ = 'teddy'
import cPickle as pickle
from time import time

from network import *
from load_data import *


num_layers = 4
num_nodes_per_layer = [[8, 8], [4, 4], [2, 2], [1, 1]]
num_cents_per_layer = [25, 25, 25, 25]
lowestLayer = 0
patch_mode = 'Adjacent'
image_type = 'Color'
# training is set true
network_mode = True
# For a Node: specify Your Algorithm Choice and Corresponding parameters
algorithm_choice = 'Clustering'
alg_params = {'mr': 0.05, 'vr': 0.05, 'sr': 0.001, 'DIMS': [], 'CENTS': [], 'node_id': [],
             'num_cents_per_layer': num_cents_per_layer}
# Declare a Network Object
DESTIN = Network(
    num_layers, algorithm_choice, alg_params, num_nodes_per_layer, patch_mode, image_type)
# training or not
DESTIN.setmode(network_mode)
DESTIN.set_lowest_layer(lowestLayer)
# Load Data
[data, labels] = loadCifar(1)
DESTIN.init_network()
# data.shape[0]
t = time()
for I in range(data.shape[0]):
    if I % 1 == 0:
        print("Training Iteration Number %d" % I)
    for L in range(DESTIN.number_of_layers):
        if L == 0:
            img = data[I][:].reshape(32, 32, 3)
            # img = 0.5*img[:,:,0] + 0.4*img[:,:,1] + 0.1*img[:,:,2]
            DESTIN.layers[0][L].load_input(img, [4, 4])
            DESTIN.layers[0][L].do_layer_learning()
        else:
            DESTIN.layers[0][L].load_input(
                DESTIN.layers[0][L - 1].nodes, [2, 2])
            DESTIN.layers[0][L].do_layer_learning()
    DESTIN.update_belief_exporter()
    if I in range(999, 50999, 500):
        Name = 'train/' + str(I + 1) + '.txt'
        file_id = open(Name, 'w')
        pickle.dump(np.array(DESTIN.network_belief['belief']), file_id)
        file_id.close()
        # Get rid-off accumulated training beliefs
        DESTIN.clean_belief_exporter()
print time() - t
exit(0)
print("Testing Started")
t = time()
network_mode = False
DESTIN.setmode(network_mode)
del data, labels
[data, labels] = loadCifar(6)
del labels
# For Every image in the data set
for I in range(data.shape[0]):
    if I % 1000 == 0:
        print("Testing Iteration Number %d" % I)
    for L in range(DESTIN.number_of_layers):
        if L == 0:
            img = data[I][:].reshape(32, 32, 3)
            DESTIN.layers[0][L].load_input(img, [4, 4])
            DESTIN.layers[0][L].do_layer_learning()
        else:
            DESTIN.layers[0][L].load_input(
                DESTIN.layers[0][L - 1].nodes, [2, 2])
            DESTIN.layers[0][L].do_layer_learning()
    DESTIN.update_belief_exporter()
    if I in range(999, 10999, 500):
        Name = 'test/' + str(I + 1) + '.txt'
        file_id = open(Name, 'w')
        pickle.dump(DESTIN.network_belief['belief'], file_id)
        file_id.close()
        # Get rid-off accumulated training beliefs
        DESTIN.clean_belief_exporter()
print time() - t

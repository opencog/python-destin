__author__ = 'teddy'
import cPickle as pickle
from Network import *
from loadData import *


"""
Here I don't move the image, I rather let a typical node move around the image
This is to make use of the methods from the previous UniformDeSTIN version
"""
# *****Define Parameters for the Network and nodes

# Network Params
num_layers = 4
patch_mode = 'Adjacent'
image_type = 'Color'
network_mode = True
# For a Node: specify Your Algorithm Choice and Corresponding parameters
'''
# ******************************************************************************************
#
#                           Incremental Clustering
#
num_nodes_per_layer = [[8, 8], [4, 4], [2, 2], [1, 1]]
num_cents_per_layer = [25, 25, 25, 25]
print "Uniform DeSTIN with Clustering"
algorithm_choice = 'Clustering'
alg_params = {'mr': 0.01, 'vr': 0.01, 'sr': 0.001, 'DIMS': [],
             'CENTS': [], 'node_id': [],
             'num_cents_per_layer': num_cents_per_layer}
# ******************************************************************************************
'''
# '''
#  ******************************************************************************************

#           Hierarchy Of AutoEncoders

print "Uniform DeSTIN with AutoEncoders"
num_nodes_per_layer = [[8, 8], [4, 4], [2, 2], [1, 1]]
num_cents_per_layer = [25, 25, 25, 25]
algorithm_choice = 'AutoEncoder'
inp_size = 48
hid_size = 100
alg_params = [[inp_size, hid_size], [4 * hid_size, hid_size],
             [4 * hid_size, hid_size], [4 * hid_size, hid_size]]
#  ******************************************************************************************
# '''

# Declare a Network Object
DESTIN = Network(
    num_layers, algorithm_choice, alg_params, num_nodes_per_layer, patch_mode, image_type)
DESTIN.setmode(network_mode)
DESTIN.set_lowest_layer(0)
# Load Data
[data, labels] = loadCifar(10)
del labels
# data = np.random.rand(5,32*32*3)
# Initialize Network; there is is also a layer-wise initialization option
DESTIN.init_network()
for I in range(data.shape[0]):  # For Every image in the data set
    if I % 200 == 0:
        print("Training Iteration Number %d" % I)
    for L in range(DESTIN.number_of_layers):
        if L == 0:
            img = data[I][:].reshape(32, 32, 3)
            DESTIN.layers[0][L].train_typical_node(img, [4, 4], algorithm_choice)
            # This is equivalent to sharing centroids or kernels
            DESTIN.layers[0][L].share_learned_parameters()
            DESTIN.layers[0][L].load_input(img, [4, 4])
            DESTIN.layers[0][L].do_layer_learning()
        else:
            DESTIN.layers[0][L].train_typical_node(
                DESTIN.layers[0][L - 1].nodes, [2, 2], algorithm_choice)
            DESTIN.layers[0][L].share_learned_parameters()
            DESTIN.layers[0][L].load_input(
                DESTIN.layers[0][L - 1].nodes, [2, 2])
            DESTIN.layers[0][L].do_layer_learning()

print("Testing Started")
network_mode = False
DESTIN.setmode(network_mode)

# On the training set
[data, labels] = loadCifar(10)
del labels
for I in range(data.shape[0]):  # For Every image in the data set
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
    if I in range(199, 50999, 200):
        Name = 'train/' + str(I + 1) + '.txt'
        file_id = open(Name, 'w')
        pickle.dump(np.array(DESTIN.network_belief['belief']), file_id)
        file_id.close()
        # Get rid-off accumulated training beliefs
        DESTIN.clean_belief_exporter()

[data, labels] = loadCifar(6)
del labels
# On the test set
for I in range(data.shape[0]):  # For Every image in the data set
    if I % 1000 == 0:
        print("Testing Iteration Number %d" % (I+50000))
    for L in range(DESTIN.number_of_layers):
        if L == 0:
            img = data[I][:].reshape(32, 32, 3)
            DESTIN.layers[0][L].load_input(img, [4, 4])
            DESTIN.layers[0][L].do_layer_learning()  # Calculates belief for
        else:
            DESTIN.layers[0][L].load_input(
                DESTIN.layers[0][L - 1].nodes, [2, 2])
            DESTIN.layers[0][L].do_layer_learning()
    DESTIN.update_belief_exporter()
    if I in range(199, 10199, 200):
        Name = 'test/' + str(I + 1) + '.txt'
        file_id = open(Name, 'w')
        pickle.dump(np.array(DESTIN.network_belief['belief']), file_id)
        file_id.close()
        # Get rid-off accumulated training beliefs
        DESTIN.clean_belief_exporter()
__author__ = 'teddy'
from Network import *
from loadData import *
from time import time
import cPickle as pickle

"""
Here I don't move the image, I rather let a typical node move around the image
This is to make use of the methods from the previous UniformDeSTIN version
"""
# *****Define Parameters for the Network and Nodes

# Network Params
numLayers = 4
PatchMode = 'Adjacent'
ImageType = 'Color'
NetworkMode = True
# For a Node: specify Your Algorithm Choice and Corresponding parameters
'''
# ******************************************************************************************
#
#                           Incremental Clustering
#
NumNodesPerLayer = [[8, 8], [4, 4], [2, 2], [1, 1]]
NumCentsPerLayer = [25, 25, 25, 25]
print "Uniform DeSTIN with Clustering"
AlgorithmChoice = 'Clustering'
AlgParams = {'mr': 0.01, 'vr': 0.01, 'sr': 0.001, 'DIMS': [],
             'CENTS': [], 'node_id': [],
             'NumCentsPerLayer': NumCentsPerLayer}
# ******************************************************************************************
'''
# '''
#  ******************************************************************************************

#           Hierarchy Of AutoEncoders

print "Uniform DeSTIN with AutoEncoders"
NumNodesPerLayer = [[8, 8], [4, 4], [2, 2], [1, 1]]
NumCentsPerLayer = [25, 25, 25, 25]
AlgorithmChoice = 'AutoEncoder'
InpSize = 48
HidSize = 100
AlgParams = [[InpSize,HidSize],[4*HidSize,HidSize],[4*HidSize,HidSize],[4*HidSize,HidSize]]
#  ******************************************************************************************
# '''

# Declare a Network Object
DESTIN = Network(
    numLayers, AlgorithmChoice, AlgParams, NumNodesPerLayer, PatchMode, ImageType)
DESTIN.setMode(NetworkMode)
DESTIN.setLowestLayer(0)
# Load Data
# data = np.random.rand(32,32,3)
[data, labels] = loadCifar(1)  # loads cifar_data_batch_1
# data = np.random.rand(5,32*32*3)
# Initialize Network; there is is also a layer-wise initialization option
DESTIN.initNetwork()
t = time()
for I in range(data.shape[0]):  # For Every image in the data set
    if I % 10 == 0:
        print time() - t
        t = time()
        print("Training Iteration Number %d" % I)
    for L in range(DESTIN.NumberOfLayers):
        if L == 0:
            img = data[I][:].reshape(32, 32, 3)
            DESTIN.Layers[0][L].trainTypicalNode(img, [4, 4], AlgorithmChoice)
            DESTIN.Layers[0][L].shareLearnedParameters()# This is equivalent to sharing centroids or kernels
            DESTIN.Layers[0][L].loadInput(img,[4,4])
            DESTIN.Layers[0][L].doLayerLearning()# Calculates belief for
            # every Node using the shared parameters and inputs of the Nodes
        else:
            DESTIN.Layers[0][L].trainTypicalNode(
                DESTIN.Layers[0][L - 1].Nodes, [2, 2], AlgorithmChoice)
            DESTIN.Layers[0][L].shareLearnedParameters()
            DESTIN.Layers[0][L].loadInput(DESTIN.Layers[0][L-1].Nodes,[2,2])
            DESTIN.Layers[0][L].doLayerLearning()
    DESTIN.updateBeliefExporter()
    if I in range(199,50999,200):
        Name = 'train/' + str(I+1) + '.txt'
        FID = open(Name,'w')
        pickle.dump(np.array(DESTIN.NetworkBelief['Belief']), FID)
        FID.close()
        DESTIN.cleanBeliefExporter()# Get rid-off accumulated training beliefs

DESTIN.dumpBelief(2)
DESTIN.cleanBeliefExporter()  # Get rid-off accumulated training beliefs
print("Testing Started")
NetworkMode = False
DESTIN.setMode(NetworkMode)
del data, labels
[data, labels] = loadCifar(6)
# data = np.random.rand(5,32*32*3)
del labels
for I in range(data.shape[0]):  # For Every image in the data set
    if I % 1000 == 0:
        print("Testing Iteration Number %d" % I)
    for L in range(DESTIN.NumberOfLayers):
        if L == 0:
            img = data[I][:].reshape(32, 32, 3)
            DESTIN.Layers[0][L].loadInput(img, [4, 4])  # loadInput to Layer[0]
        else:
            DESTIN.Layers[0][L].loadInput(
                DESTIN.Layers[0][L - 1].Nodes, [2, 2])
        # Only belief calculation NoTraining
        DESTIN.Layers[0][L].doLayerLearning()
    DESTIN.updateBeliefExporter()
DESTIN.dumpBelief(2)

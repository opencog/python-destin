__author__ = 'teddy'
from Network import *
from loadData import *
print("Uniform DeSTIN")

"""
Here I don't move the image, I rather let the nodes move around the image
This is to make use of the methods from the previous UniformDeSTIN version
"""
# *****Define Parameters for the Network and Nodes

#Network Params
numLayers = 4
NumNodesPerLayer = [[8, 8], [4, 4], [2, 2], [1, 1]]
NumCentsPerLayer = [25, 25, 25, 25]
PatchMode = 'Adjacent'  #
ImageType = 'Color'
NetworkMode = True # training is set true
#For a Node: specify Your Algorithm Choice and Corresponding parameters
AlgorithmChoice = 'Clustering'
AlgParams = {'mr': 0.01, 'vr': 0.01, 'sr': 0.001, 'DIMS': [], 'CENTS': [], 'node_id': [],
             'NumCentsPerLayer': NumCentsPerLayer}
#Declare a Network Object
DESTIN = Network(numLayers, AlgorithmChoice, AlgParams, NumNodesPerLayer, PatchMode, ImageType)
DESTIN.setMode(NetworkMode) #training or not
DESTIN.setLowestLayer(0)
#Load Data
[data, labels] = loadCifar(10) # loads cifar_data_batch_1
#data = np.random.rand(5,32*32*3)
#Initialize Network; there is is also a layer-wise initialization option
DESTIN.initNetwork()
#data.shape[0]
for I in range(data.shape[0]):# For Every image in the data set
    if I%1000 == 0:
        print("Training Iteration Number %d" % I)
    for L in range(DESTIN.NumberOfLayers):
        if L == 0:
            img = data[I][:].reshape(32,32,3)
            DESTIN.Layers[0][L].trainTypicalNode(img,[4,4])
        else:
            DESTIN.Layers[0][L].trainTypicalNode(DESTIN.Layers[0][L-1].Nodes,[2,2])
        DESTIN.Layers[0][L].shareCentroids()
    DESTIN.updateBeliefExporter()
DESTIN.dumpBelief(2)
DESTIN.cleanBeliefExporter()#Get rid-off accumulated training beliefs
print("Testing Started")
NetworkMode = False
DESTIN.setMode(NetworkMode)
del data, labels
[data, labels] = loadCifar(6)
#data = np.random.rand(5,32*32*3)
del labels
for I in range(data.shape[0]):# For Every image in the data set
    if I%1000 == 0:
        print("Testing Iteration Number %d" % I)
    for L in range(DESTIN.NumberOfLayers):
        if L == 0:
            img = data[I][:].reshape(32,32,3)
            DESTIN.Layers[0][L].loadInput(img,[4,4])# loadInput to Layer[0]
        else:
            DESTIN.Layers[0][L].loadInput(DESTIN.Layers[0][L-1].Nodes,[2,2])
        DESTIN.Layers[0][L].doLayerLearning()# Only belief calculation NoTraining
    DESTIN.updateBeliefExporter()
DESTIN.dumpBelief(2)
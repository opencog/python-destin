__author__ = 'teddy'
from Network import *
from loadData import *
import cPickle as pickle
from time import time

numLayers = 4
NumNodesPerLayer = [[8, 8], [4, 4], [2, 2], [1, 1]]
NumCentsPerLayer = [25, 25, 25, 25]
lowestLayer = 0
PatchMode = 'Adjacent'  #
ImageType = 'Color'
NetworkMode = True # training is set true
#For a Node: specify Your Algorithm Choice and Corresponding parameters
AlgorithmChoice = 'Clustering'
AlgParams = {'mr': 0.05, 'vr': 0.05, 'sr': 0.001, 'DIMS': [], 'CENTS': [], 'node_id': [],
             'NumCentsPerLayer': NumCentsPerLayer}
#Declare a Network Object
DESTIN = Network(numLayers, AlgorithmChoice, AlgParams, NumNodesPerLayer, PatchMode, ImageType)
DESTIN.setMode(NetworkMode) #training or not
DESTIN.setLowestLayer(lowestLayer)
#Load Data
[data, labels] = loadCifar(1) # loads cifar_data_batch_1
#Initialize Network; there is is also a layer-wise initialization option
DESTIN.initNetwork()
#data.shape[0]
t = time()
#ret = load_cifar()
#DESTIN.network_init_whitening(ret['patch_mean'], ret['patch_std'], ret['whiten_mat'])
for I in range(data.shape[0]):
    if I%1 == 0:
        print("Training Iteration Number %d" % I)
    for L in range(DESTIN.NumberOfLayers):
        if L == 0:
            img = data[I][:].reshape(32,32,3)
            #img = 0.5*img[:,:,0] + 0.4*img[:,:,1] + 0.1*img[:,:,2]
            DESTIN.Layers[0][L].loadInput(img,[4,4])
            DESTIN.Layers[0][L].doLayerLearning()
        else:
            DESTIN.Layers[0][L].loadInput(DESTIN.Layers[0][L-1].Nodes,[2,2])
            DESTIN.Layers[0][L].doLayerLearning()
    DESTIN.updateBeliefExporter()
    if I in range(999,50999,500):
        Name = 'train/' + str(I+1) + '.txt'
        FID = open(Name,'w')
        pickle.dump(np.array(DESTIN.NetworkBelief['Belief']),FID)
        FID.close()
        DESTIN.cleanBeliefExporter()#Get rid-off accumulated training beliefs
print time() - t
exit(0)
print("Testing Started")
t = time()
NetworkMode = False
DESTIN.setMode(NetworkMode)
del data, labels
[data,labels] = loadCifar(6)
del labels
for I in range(data.shape[0]):# For Every image in the data set
    if I%1000 == 0:
        print("Testing Iteration Number %d" % I)
    for L in range(DESTIN.NumberOfLayers):
        if L == 0:
            img = data[I][:].reshape(32,32,3)
            DESTIN.Layers[0][L].loadInput(img,[4,4])# loadInput to Layer[0]
            DESTIN.Layers[0][L].doLayerLearning()
        else:
            DESTIN.Layers[0][L].loadInput(DESTIN.Layers[0][L-1].Nodes,[2,2])
            DESTIN.Layers[0][L].doLayerLearning()
    DESTIN.updateBeliefExporter()
    if I in range(999,10999,500):
        Name = 'test/' + str(I+1) + '.txt'
        FID = open(Name,'w')
        pickle.dump(DESTIN.NetworkBelief['Belief'],FID)
        FID.close()
        DESTIN.cleanBeliefExporter()#Get rid-off accumulated training beliefs
print time() - t

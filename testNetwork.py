import numpy.random as rng
import numpy as np
import theano
from Network import *
from Node import *

# self,numLayers,AlgChoice,AlgParams,NumNodesPerLayer,PatchMode='Adjacent',ImageType='Color'
numLayers = 4
NumNodesPerLayer = [[8, 8], [4, 4], [2, 2], [1, 1]]
<<<<<<< HEAD
# Here AlgParams are being initialized
=======
# # Here AlgParams are being initialized
>>>>>>> ab02f8d2a9e47bf672c5d77c5f60b408df2c9fdb
N = 1
feats = 16
img = np.random.rand(32, 32)
Label = 1
Ratio = 4
AlgorithmChoice = 'LogRegression'
AlgParams = {}
<<<<<<< HEAD
# AlgParams['N'] =
=======
#AlgParams['N'] =
>>>>>>> ab02f8d2a9e47bf672c5d77c5f60b408df2c9fdb
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
AlgParams['D'] = D  # an initial random input
AlgParams['N'] = N
AlgParams['feats'] = feats
AlgParams['training_steps'] = 1
w = theano.shared(rng.randn(feats), name="w")
AlgParams['w'] = w
<<<<<<< HEAD
myNetwork = Network(
    numLayers, AlgorithmChoice, AlgParams, NumNodesPerLayer, 'Adjacent', 'Gray')
=======
myNetwork = Network(numLayers, AlgorithmChoice, AlgParams, NumNodesPerLayer, 'Adjacent', 'Gray')
>>>>>>> ab02f8d2a9e47bf672c5d77c5f60b408df2c9fdb
myNetwork.Layers[0][0].loadInput(img, 4)
myNetwork.Layers[0][0].initLayerLearningParams('LogRegression', AlgParams)
myNetwork.Layers[0][0].doLayerLearning(1)
exit(0)
for I in range(len(myNetwork.Layers)):
    for J in range(len(myNetwork.Layers[0])):
        if J == 0:
            myNetwork.Layers[I][J].loadInput(img, 4)
        else:
            print type(J - 1)
            exit(0)
<<<<<<< HEAD
            myNetwork.Layers[I][0].loadInput(
                myNetwork.Layers[I][J - 1].Nodes, 4)
        myNetwork.Layers[I][0].loadInput(img, 4)
        myNetwork.initLayer(0)
        # myNetwork.trainLayer(0)
=======
            myNetwork.Layers[I][0].loadInput(myNetwork.Layers[I][J - 1].Nodes, 4)
        myNetwork.Layers[I][0].loadInput(img, 4)
        myNetwork.initLayer(0)
        #myNetwork.trainLayer(0)
>>>>>>> ab02f8d2a9e47bf672c5d77c5f60b408df2c9fdb
'''
exit(0)
myNetwork.Layers[0][0].loadInput(img, 4)
myNetwork.initLayer(0)
myNetwork.trainLayer(0)
'''
'''
print "number of Layers" 
print [len(myNetwork.Layers),len(myNetwork.Layers[0])]
print "number of Nodes in Layer 1" 
print [myNetwork.Layers[0][0].Nodes[7][7].AlgorithmChoice]
'''

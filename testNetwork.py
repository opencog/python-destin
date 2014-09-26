import numpy.random as rng
import numpy as np
import theano
from Network import *
from Node import *

# self,num_layers,alg_choice,alg_params,num_nodes_per_layer,patch_mode='Adjacent',image_type='Color'
num_layers = 4
num_nodes_per_layer = [[8, 8], [4, 4], [2, 2], [1, 1]]
# Here alg_params are being initialized
N = 1
feats = 16
img = np.random.rand(32, 32)
Label = 1
Ratio = 4
algorithm_choice = 'LogRegression'
alg_params = {}

# alg_params['N'] =
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
alg_params['D'] = D  # an initial random input
alg_params['N'] = N
alg_params['feats'] = feats
alg_params['training_steps'] = 1
w = theano.shared(rng.randn(feats), name="w")
alg_params['w'] = w
myNetwork = Network(
    num_layers, algorithm_choice, alg_params, num_nodes_per_layer, 'Adjacent', 'Gray')
myNetwork.layers[0][0].load_input(img, 4)
myNetwork.layers[0][0].init_layer_learning_params('LogRegression', alg_params)
myNetwork.layers[0][0].do_layer_learning(1)
exit(0)
for I in range(len(myNetwork.layers)):
    for J in range(len(myNetwork.layers[0])):
        if J == 0:
            myNetwork.layers[I][J].load_input(img, 4)
        else:
            print type(J - 1)
            exit(0)
            myNetwork.layers[I][0].load_input(
                myNetwork.layers[I][J - 1].nodes, 4)
        myNetwork.layers[I][0].load_input(img, 4)
        myNetwork.initLayer(0)
        # myNetwork.train_layer(0)
'''
exit(0)
myNetwork.layers[0][0].load_input(img, 4)
myNetwork.initLayer(0)
myNetwork.train_layer(0)
'''
'''
print "number of layers"
print [len(myNetwork.layers),len(myNetwork.layers[0])]
print "number of nodes in Layer 1"
print [myNetwork.layers[0][0].nodes[7][7].algorithm_choice]
'''

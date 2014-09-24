__author__ = 'teddy'
from AutoEncoder import *
import scipy.io as io
from pprint import pprint
# declaring an AE object by specifying the network dimensions let inpDim =
# 4 hidDim = 4
testAE = NNSAE(400, 200)
epoch = 1
Data = io.loadmat('data.mat')
y = Data['y']
X = Data['X']
for I in range(X.shape[0]):
    if I % 100 == 0:
        print I
    for Temp in range(0, epoch, 1):
        testAE.train((X[I][:].reshape(1, 400)))
print np.shape(testAE.W)
D1 = np.asarray(testAE.apply((X[100][:].reshape(1, 400))))
D2 = np.asarray(testAE.apply((X[101][:].reshape(1, 400))))

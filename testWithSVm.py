__author__ = 'teddy'
# imports
import cPickle as pickle

from sklearn import svm

from loadData import *

import scipy.io as io

# Loading Training Data and Labels
temp = np.array([])
cifarDir = '/home/teddy/Desktop/destin/NonUniform/PythonDeSTIN/cifar-10-batches-mat'
Name = cifarDir + '/data_batch_1.mat'
temp1 = io.loadmat(Name)
temp = np.array(temp1['labels'])

for I in range(2, 6):
    Name = cifarDir + '/data_batch_' + str(I) + '.mat'
    temp1 = io.loadmat(Name)
    temp2 = np.array(temp1['labels'])
    #print temp2.shape
    #print temp.shape
    temp = np.vstack((temp, temp2))

del temp1
del temp2
trainLabel = np.array(temp)
#del temp, temp1, temp2
temp = io.loadmat(cifarDir + '/test_batch' + '.mat')
testLabel = np.array(temp['labels'])
del temp
A = np.array([])
for I in range(499, 50000, 500):
    Name = 'train/' + str(I + 1) + '.txt'
    FID = open(Name, 'r')
    Temp = np.array(pickle.load(FID))
    FID.close()
    A = np.hstack((A, Temp))
del Temp
totLen = len(A)
Width = int(totLen / 50000)
A = A.reshape(50000, Width)

#Training SVM
SVM = svm.SVC(C=100, kernel='rbf')
print "Fitting The SVM samples"
trainLabel = np.squeeze(np.asarray(trainLabel).reshape(50000, 1))
SVM.fit(A, trainLabel)
trainPred = SVM.predict(A)

trainAccuracy = float(sum(trainPred == trainLabel)) / float(len(trainLabel))
print("Training Accuracy = %f" % trainAccuracy*100)
del A
#
A2 = np.array([])
for I in range(499, 10000, 500):
    Name = 'test/' + str(I + 1) + '.txt'
    FID = open(Name, 'r')
    Temp = np.array(pickle.load(FID))
    FID.close()
    A2 = np.hstack((A2, Temp))
totLen = len(A2)
Width = int(totLen / 10000)
A2 = A2.reshape(10000, Width)
#A = A[0:10000,1599:2124]
del Temp
print "Predicting Test samples"
testPred = SVM.predict(A2)
del A2
[testData, testLabel] = loadCifar(6)
del testData
testLabel = np.array(testLabel).reshape(10000, 1)
testLabel = np.squeeze(np.asarray(testLabel).reshape(10000, 1))
testAccuracy = float(sum(testPred == testLabel)) / float(len(testLabel))
#print testPred[0:400]
#print testLabel[0:400]
print("Training Accuracy = %f" % testAccuracy*100)


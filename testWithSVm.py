__author__ = 'teddy'
# imports
import cPickle as pickle
from sklearn import svm
from loadData import *

# Loading Training and Testing Labels from Cifar data set
[trainData, trainLabel] = loadCifar(10)
del trainData
[testData, testLabel] = loadCifar(6)
del testData

# Load Training and Test Data/Extracted from DeSTIN
trainData = np.array([])
for I in range(199, 50000, 200):
    Name = 'train/' + str(I + 1) + '.txt'
    FID = open(Name, 'r')
    Temp = np.array(pickle.load(FID))
    FID.close()
    trainData = np.hstack((trainData, Temp))
del Temp
totLen = len(trainData)
Width = int(totLen / 50000)
trainData = trainData.reshape(50000, Width)

# Training SVM
SVM = svm.LinearSVC(C=100, kernel='rbf')
print "Fitting The SVM samples"
trainLabel = np.squeeze(np.asarray(trainLabel).reshape(50000, 1))
SVM.fit(trainData, trainLabel)

print("Training Accuracy = %f" % SVM.score((trainData,trainLabel) * 100))
del trainData
#
testData = np.array([])
for I in range(199, 10000, 200):
    Name = 'test/' + str(I + 1) + '.txt'
    FID = open(Name, 'r')
    Temp = np.array(pickle.load(FID))
    FID.close()
    testData = np.hstack((testData, Temp))
totLen = len(testData)
Width = int(totLen / 10000)
testData = testData.reshape(10000, Width)
# A = A[0:10000,1599:2124]
del Temp
print "Predicting Test samples"
print("Training Accuracy = %f" % (SVM.score(testData, testLabel) * 100))

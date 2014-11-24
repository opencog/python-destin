__author__ = 'teddy'
# imports
import cPickle as pickle
import scipy.io as io

from sklearn import svm

from load_data import *


# Loading Training and Testing Labels from cifar data set
print("Loading training and test labels")
[trainData, trainLabel] = loadCifar(10)
del trainData
[testData, testLabel] = loadCifar(6)
del testData

# Load Training and Test Data/Extracted from DeSTIN
print("Loading training and testing features")
trainData = np.array([])
for I in range(199, 50000, 200):
    Name = 'train/' + str(I + 1) + '.txt'
    file_id = open(Name, 'r')
    Temp = np.array(pickle.load(file_id))
    file_id.close()
    trainData = np.hstack((trainData, Temp))
del Temp
totLen = len(trainData)
Width = int(totLen / 50000)
trainData = trainData.reshape(50000, Width)
trainData = trainData[:, 6400:8500]

# Training SVM
SVM = svm.LinearSVC(C=10)
# C=100, kernel='rbf')
print "Training the SVM"
trainLabel = np.squeeze(np.asarray(trainLabel).reshape(50000, 1))
SVM.fit(trainData, trainLabel)
print("Training Score = %f " % float(100 * SVM.score(trainData, trainLabel, sample_weight=None)))
#print("Training Accuracy = %f" % (SVM.score(trainData, trainLabel) * 100))
eff = {}
eff['train'] = SVM.score(trainData, trainLabel) * 100
del trainData

testData = np.array([])
for I in range(199, 10000, 200):
    Name = 'test/' + str(I + 1) + '.txt'
    file_id = open(Name, 'r')
    Temp = np.array(pickle.load(file_id))
    file_id.close()
    testData = np.hstack((testData, Temp))
totLen = len(testData)
Width = int(totLen / 10000)
testData = testData.reshape(10000, Width)
testData = testData[:, 6400:8500]
del Temp
print "Predicting Test samples"
print("Test Score = %f" % float(100 * SVM.score(testData, testLabel, sample_weight = None)))
#print("Training Accuracy = %f" % (SVM.score(testData, testLabel) * 100))
eff['test'] = SVM.score(testData, testLabel) * 100
io.savemat('accuracy.mat', eff)
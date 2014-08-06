# -*- coding: utf-8 -*-
__author__ = 'teddy'
import scipy.io as io

from Layer import *

# io.savemat(FileName,Dict,True)
# TODO: get ridoff the sequential requirements like first feed the layer an input the you can initialize it

class Network():
    def __init__(self, numLayers, AlgChoice, AlgParams, NumNodesPerLayer, PatchMode='Adjacent', ImageType='Color'):
        self.NetworkBelief = {}
        self.LowestLayer = 1
        self.NetworkBelief['Belief'] = np.array([])  # this is going to store beliefs for every image DeSTIN sees
        self.saveBeliefOption = 'True'
        self.BeliefFileName = 'Beliefs.mat'
        self.NumberOfLayers = numLayers
        self.AlgorithmChoice = AlgChoice
        self.AlgorithmParams = AlgParams
        self.NumberOfNodesPerLayer = NumNodesPerLayer
        self.PatchMode = PatchMode
        self.ImageType = ImageType
        self.Layers = [[Layer(j, NumNodesPerLayer[j], self.PatchMode, self.ImageType) for j in range(numLayers)]]

    def setMode(self, Mode):
        self.OperatingMode = Mode
        for I in range(self.NumberOfLayers):
            self.Layers[0][I].Mode = Mode
    def initNetwork(self):
        for L in range(self.NumberOfLayers):
            self.initLayer(L)
    def setLowestLayer(self,LowestLayer):
        self.LowestLayer = LowestLayer

    def initLayer(self, LayerNum):  # TODO make sure lower layer is initialized (or trained at least once)
        self.Layers[0][LayerNum].initLayerLearningParams(self.AlgorithmChoice, self.AlgorithmParams)

    def trainLayer(self, LayerNum):
        self.Layers[0][LayerNum].doLayerLearning(self.OperatingMode)

    def updateBeliefExporter(self):
        for i in range(self.LowestLayer, self.NumberOfLayers):
            for j in range(len(self.Layers[0][i].Nodes)):
                for k in range(len(self.Layers[0][i].Nodes[0])):
                    if self.NetworkBelief['Belief'] == np.array([]):
                        self.NetworkBelief['Belief'] = np.array(self.Layers[0][i].Nodes[j][k].Belief).ravel()
                    else:
                        self.NetworkBelief['Belief'] = np.hstack((np.array(self.NetworkBelief['Belief']),
                                                              np.array(self.Layers[0][i].Nodes[j][k].Belief).ravel()))

    def dumpBelief(self, NumOfImages):
        TotalBeliefLen = len(np.array(self.NetworkBelief).ravel())
        SingleBeliefLen = TotalBeliefLen/NumOfImages
        print np.array(self.NetworkBelief).ravel()
        Belief = np.array(self.NetworkBelief).reshape(NumOfImages,SingleBeliefLen)
        io.savemat(self.BeliefFileName, Belief)
    def cleanBeliefExporter(self):
        self.NetworkBelief['Belief'] = np.array([])
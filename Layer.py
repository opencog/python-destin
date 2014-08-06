# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 2014
@author: teddy
"""
from Node import *
from loadData import *
class Layer:
    def __init__(self, LayerNum, NumberOfNodes, PatchMode=None, ImageType=None):
        self.PatchMode = PatchMode
        self.ImageType = ImageType
        self.LayerNumber = LayerNum
        self.NumberOfNodes = NumberOfNodes  # Usually a list with two elements
        Row = NumberOfNodes[0]
        Col = NumberOfNodes[1]
        if LayerNum == 0:
            Nodes = [[Node(LayerNum, [i, j], load_cifar(4)) for j in range(Row)] for i in range(Col)]
        else:
            Nodes = [[Node(LayerNum, [i, j]) for j in range(Row)] for i in range(Col)]
        self.Nodes = Nodes
        self.Mode = []

    def loadInput(self, Input, R):
        Ratio = R[0]
        # Ratio equals to the number of lower layer units getting combined and being fed to the upper layer
        if self.LayerNumber == 0:
            Nx = 0  # X coordinate of the current node
            for I in range(0, Input.shape[0], Ratio):
                Ny = 0  # Y coordinate of the current node
                for J in range(0, Input.shape[1], Ratio):
                    self.Nodes[Nx][Ny].loadInput(returnNodeInput(Input, [I, J], Ratio, self.PatchMode,
                                                                 self.ImageType))  # returns inputs to the node located at Position [Nx,Ny]
                    Ny += 1
                Nx += 1
        else:
            Nx = 0  # X coordinate of the current node
            Ny = 0  # Y coordinate of the current node
            for I in range(0, len(Input[0]), Ratio):
                Ny = 0
                for J in range(0, len(Input[1]), Ratio):
                    InputTemp = np.array([])
                    for K in range(I, I + Ratio):
                        for L in range(J, J + Ratio):
                            InputTemp = np.append(InputTemp, np.asarray(Input[K][
                                                                            L].Belief))# Combine the Beliefs of the Nodes passed
                    self.Nodes[Nx][Ny].loadInput(np.ravel(InputTemp))
                    Ny += 1
                Nx += 1

    def initLayerLearningParams(self, AlgorithmChoice, AlgParams):
        for I in range(len(self.Nodes)):
            for J in range(len(self.Nodes[0])):
                self.Nodes[I][J].initNodeLearningParams(AlgorithmChoice, AlgParams)

    def doLayerLearning(self):
        for I in range(len(self.Nodes)):
            for J in range(len(self.Nodes[0])):
                self.Nodes[I][J].doNodeLearning(self.Mode)
    def trainTypicalNode(self,Input,windowSize):
        TN = self.Nodes[0][0]
        [H,V] = windowSize
        if self.LayerNumber == 0:
            X = Input.shape[0] - H + 1
            Y = Input.shape[1] - V + 1
            for I in range(X):
                for J in range(Y):
                    TN.loadInput(returnNodeInput(Input, [I, J], H, self.PatchMode, self.ImageType))
                    TN.doNodeLearning(self.Mode)
        else:
            X = len(Input[0]) - H + 1
            Y = len(Input[1]) - V + 1
            for I in range(X):
                for J in range(Y):
                    InputTemp = np.array([])
                    for K in range(I, I+H):
                        for L in range(J, J+V):
                            InputTemp = np.append(InputTemp, np.array(np.ravel(Input[K][L].Belief)))
                            # Combine the Beliefs of the Nodes passed
                    TN.loadInput(np.ravel(InputTemp))
                    TN.doNodeLearning(self.Nodes)
        self.Nodes[0][0] = TN

    def shareCentroids(self):
        for I in range(len(self.Nodes)):
            for J in range(len(self.Nodes[0])):
                self.Nodes[I][J] = self.Nodes[0][0]
    def updateBeliefs(self):
        for I in range(len(self.Nodes)):
            for J in range(len(self.Nodes[0])):
                self.Nodes[I][J].doNodeLearning(False)
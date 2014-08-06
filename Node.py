# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 2014
@author: teddy
"""

from Clustering import *

class Node:
    def __init__(self, LayerNumber, NodePos, cifarstat={'patch_mean':[],'patch_std':[],'whiten_mat':[]}):
        self.LayerNumber = LayerNumber
        self.NodePosition = NodePos
        self.Belief = []
        #cifarStat = load_cifar(4)# to be used for Normalization and Whitening Purposes
        self.patch_mean = cifarstat['patch_mean']
        self.patch_std = cifarstat['patch_std']
        self.v = cifarstat['whiten_mat']

    def initNodeLearningParams(self, AlgorithmChoice, AlgParams):
        self.AlgorithmChoice = AlgorithmChoice
        if AlgorithmChoice == 'Clustering':
            CentsPerLayer = AlgParams['NumCentsPerLayer']
            # InputWidth = InputWidths[LayerNum]
            if self.LayerNumber == 0:
                InputWidth = 48
            else:
                InputWidth = CentsPerLayer[self.LayerNumber-1] * 4
            self.LearningAlgorithm = Clustering(AlgParams['mr'], AlgParams['vr'], AlgParams['sr'], InputWidth,
                                                AlgParams['NumCentsPerLayer'][self.LayerNumber], self.NodePosition)
        else:
            print('Only Incremental Clustering Exists')

    def loadInput(self, In):
        if self.LayerNumber == 0:
            In = In - self.patch_mean
            In = In/self.patch_std
            In = In.dot(self.v)
        self.Input = In

    def doNodeLearning(self, Mode):
        if self.AlgorithmChoice == 'Clustering':
            self.LearningAlgorithm.update_node(self.Input, Mode)
            self.Belief = self.LearningAlgorithm.belief
        else:
            print("Only Incremental Clustering Algorithm Exists")
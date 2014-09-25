#  -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 2014
@author: teddy
"""

from Clustering import *
from AutoEncoder import *


class Node:

    def __init__(self, LayerNumber, NodePos, cifarstat={'patch_mean': [], 'patch_std': [], 'whiten_mat': []}):
        self.LayerNumber = LayerNumber
        self.NodePosition = NodePos
        self.Belief = []
        # cifarStat = load_cifar(4)#  to be used for Normalization and Whitening
        #  Purposes
        self.patch_mean = cifarstat['patch_mean']
        self.patch_std = cifarstat['patch_std']
        self.v = cifarstat['whiten_mat']

    def initNodeLearningParams(self, AlgorithmChoice, AlgParams):
        self.AlgorithmChoice = AlgorithmChoice
        if AlgorithmChoice == 'Clustering':
            CentsPerLayer = AlgParams['NumCentsPerLayer']
            #  InputWidth = InputWidths[LayerNum]
            if self.LayerNumber == 0:
                InputWidth = 48
            else:
                InputWidth = CentsPerLayer[self.LayerNumber - 1] * 4
            self.LearningAlgorithm = Clustering(AlgParams['mr'], AlgParams['vr'], AlgParams['sr'], InputWidth,
                                                AlgParams['NumCentsPerLayer'][self.LayerNumber], self.NodePosition)
            # mr, vr, sr, di, ce, node_id
        else:
            self.Belief = np.ones((AlgParams[self.LayerNumber][1], 1))
            self.AlgorithmChoice = AlgorithmChoice
            self.LearningAlgorithm = NNSAE(
                AlgParams[self.LayerNumber][0], AlgParams[self.LayerNumber][1])

    def loadInput(self, In):
        if self.LayerNumber == 0:
            In = In - self.patch_mean
            In = In / self.patch_std
            In = In.dot(self.v)
        self.Input = In

    def doNodeLearning(self, Mode):
        if self.AlgorithmChoice == 'Clustering':
            self.LearningAlgorithm.update_node(self.Input, Mode)
            self.Belief = self.LearningAlgorithm.belief
        else:
            self.LearningAlgorithm.train(self.Input)
            Activations = np.dot(
                np.transpose(self.LearningAlgorithm.W), np.transpose(self.Input))
            Activations = Activations / (np.sum(Activations))
            Belief = self.Belief
            Belief = np.maximum(Activations, 0)
            self.Belief = Belief
            #  for K in range(Activations.shape[0]):
            #     Belief[K] = max(0.0, float((Activations[K] - 0.025)))
            self.Belief = np.asarray(Belief)

            """

    def calcuateBelief(self, Input):
        self.loadInputToNodes(Input, [4,4])
        for I in range(len(self.Nodes)):
            for J in range(len(self.Nodes[0])):
                W = np.transpose(self.NNSAE.W)
                Image = returnNodeInput(Input, [I*4, J*4], [4,4], 'Adjacent', 'Color')
                Activations = np.dot(W, np.transpose(Image))
                Activations = Activations/sum(sum(Activations))
                self.Nodes[I][J].Activation = Activations
                m = np.mean(np.mean(Activations,1))
                for K in range(Activations.shape[0]):
                        self.Nodes[I][J].Belief[K,0] = max(0, (Activations[K,0] - 0.025))
                print self.Nodes[0][0].Belief"""

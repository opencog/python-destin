__author__ = 'teddy'
import numpy as np
import numpy.random as rand


class NNSAE:

    def __init__(self, inpDim, hidDim):
        self.inpDim = inpDim  #number of input neurons (and output neurons)
        self.hidDim = hidDim  #number of hidden neurons
        self.inp = np.zeros((inpDim, 1))  #vector holding current input
        self.out = np.zeros((inpDim, 1))  #output neurons
        self.g = np.zeros((inpDim, 1))  #neural activity before non-linearity
        self.a = np.ones((hidDim, 1))
        self.h = np.zeros((inpDim, 1))  #hidden neuron activation
        self.b = -3 * np.ones((hidDim, 1))
        self.lrateRO = 0.01  #learning rate for synaptic plasticity of read-out layer (RO)
        self.regRO = 0.0002#0.0001 * (2 / (3 * lnum))  #numerical regularization constant
        self.decayP = 0  #decay factor for positive weights [0..1]
        self.decayN = 1  #decay factor for negative weights [0..1]
        self.lrateIP = 0.001  #learning rate for intrinsic plasticity (IP)
        self.meanIP = 0.2  #desired mean activity, a parameter of IP
        self.W = 0.025 * (2 * rand.rand(inpDim, hidDim) - 0.5 * np.ones((inpDim, hidDim))) + 0.025;

    def apply(self, X):
        XNew = np.asarray(X).transpose()
        if np.asarray(XNew).shape[1] != 1:
            print('Use Input which are Row Vectors of shape (1xL)')
            #exit(1)
        else:
            self.inp = XNew
            self.update()
            Xhat = self.out
            return Xhat

    def train(self, X):
        #set Input
        self.inp = np.asarray(X).transpose()
        if np.asarray(self.inp).shape[1] != 1:
            print('Use Inputs which are Row Vectors of shape (1xL)')
            #exit(1)
        # do forward propagation of activities
        self.update()

        # calculate adaptive learning rate
        lrate = self.lrateRO / (self.regRO + sum(np.power(self.h, 2)))

        #calculate erros
        error = self.inp - self.out

        #update weights
        self.W = self.W + lrate * error * self.h.transpose()

        #decay function for positive weights
        if self.decayP > 0:
            idx = np.where(self.W > 0)
            idx0 = list(idx[0])
            idx1 = list(idx[1])
            Len = len(idx[0])
            if Len > 0:
                for I in range(Len):
                    #net.W(idx) = net.W(idx) - net.decayP * net.W(idx)
                    self.W[idx0[I]][idx1[I]] = self.W[idx0[I]][idx1[I]] - self.decayP * self.W[idx0[I]][idx1[I]]
        #decay functions for negative weights
        if self.decayN == 1:
            self.W = np.maximum(self.W, 0)
        elif self.decayN > 0:
            idx = np.where(self.W < 0)
            idx0 = list(idx[0])
            idx1 = list(idx[1])
            Len = len(idx[0])
            for I in range(Len):
                self.W[idx0[I]][idx1[I]] = self.W[idx0[I]][idx1[I]] - self.decayN * self.W[idx0[I]][idx1[I]]
        else:
            pass
        #Intrinsic Plasticity
        hones = np.ones((self.hidDim,1))
        tmp = self.lrateIP * (hones - (2.0 + float(1.0)/self.meanIP) * self.h + np.power(self.h, 2)/self.meanIP)
        self.b = self.b + tmp
        self.a = self.a + self.lrateIP * (hones / (self.a) + self.g * tmp)
    def update(self):
        self.g = np.dot(self.W.transpose(), self.inp)
        #Apply activation function
        self.h = float(1)/(1 + np.exp(-self.a * self.g - self.b))

        #Read-out
        self.out =  np.dot(self.W, self.h)

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 2014
@author: teddy
"""

#import theano
#import theano.tensor as T

'''
def LogReg(D, N, training_steps, feats, w):
    # Declare Theano symbolic variables
    x = T.matrix("x")
    y = T.vector("y")
    b = theano.shared(0., name="b")
    print "Initial model:"
    print w.get_value(), b.get_value()

    # Construct Theano expression graph
    p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))  # Probability that target = 1
    prediction = p_1 > 0.5  # The prediction thresholded
    xent = -y * T.log(p_1) - (1 - y) * T.log(1 - p_1)  # Cross-entropy loss function
    cost = xent.mean() + 0.01 * (w ** 2).sum()  # The cost to minimize
    gw, gb = T.grad(cost, [w, b])  # Compute the gradient of the cost

    # Compile
    train = theano.function(
        inputs=[x, y],
        outputs=[prediction, xent],
        updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
    predict = theano.function(inputs=[x], outputs=prediction)

    # Train
    for i in range(training_steps):
        pred, err = train(D[0], D[1])
'''

class LearningAlgorithm:
    def __init__(self, AlgParams):
        self.D = AlgParams['D']
        self.N = AlgParams['N']
        self.training_steps = AlgParams['training_steps']
        self.feats = AlgParams['feats']
        self.w = AlgParams['w']

    def runLearningAlgorithm(self, Mode):
        LogReg(self.D, self.N, self.training_steps, self.feats, self.w)

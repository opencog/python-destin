__author__ = 'Steven'
# Re-implemented in Theano at iCogLabs
import numpy as np
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
from elementaryTheanoFunctions import *


class Clustering:

    """
    This is the basic clustering class.
    """

    def __init__(self, mr, vr, sr, di, ce, node_id):
        """
        Initializes the clustering class.
        """

        self.common_init(mr, vr, sr, di, ce, node_id)

    def common_init(self, mr, vr, sr, di, ce, node_id):
        """
        Initialization function used by the base class and subclasses.
        """

        self.MEANRATE = mr
        self.VARRATE = vr
        self.STARVRATE = sr
        self.DIMS = di
        self.CENTS = ce
        self.ID = node_id
        srng = RandomStreams(seed=100)
        rv_u = srng.uniform((self.CENTS, self.DIMS))
        f = function([], rv_u)
        self.mean = 2 * f()
        # print self.mean
        var1 = T.dscalar('var1')
        var2 = T.dmatrix('var2')
        var3 = T.mul
        self.var = theanoScaMatMul(0.001, np.ones((self.CENTS, self.DIMS)))
        self.starv = np.ones((self.CENTS, 1))
        self.belief = np.zeros((1, self.CENTS))
        self.children = []
        self.last = np.zeros((1, self.DIMS))
        self.whitening = False

    def update_node(self, input, TRAIN):
        """
        Update the node based on an input and training flag.
        """

        # print self.DIMS
        # print input.shape[0]
        input = input.reshape(1, self.DIMS)
        self.process_input(input, TRAIN)

    def process_input(self, input, TRAIN):
        """
        Node update function for base class and subclasses.
        """

        # Calculate Distance
        diff = input - self.mean
        sqdiff = np.square(diff)
        if TRAIN:
            self.train_node(diff, sqdiff)
        self.produce_belief(sqdiff)

    def train_node(self, diff, sqdiff):
        """
        Node model (means, variances) update function for base
        class and subclasses. Subclass should overide if
        Euclidean distance is not desired for selecting
        winning centroid.
        """
        var1 = T.dmatrix('var1')
        SQRT = function([var1], [T.sqrt(var1)])
        SUM = function([var1], [T.sum(var1, axis=1)])
        euc = np.array(SQRT(SUM(sqdiff))).reshape(self.CENTS, 1)
        self.update_winner(euc, diff)

    def update_winner(self, dist, diff):
        """
        Updates winning centroid. Subclasses should not
        need to overide this function.
        """

        # Apply starvation trace
        dist = dist * self.starv

        # Find and Update Winner
        winner = np.argmin(dist)
        self.mean[winner, :] += self.MEANRATE * diff[winner, :]
        # this should be updated to use sqdiff
        vdiff = np.square(diff[winner, :]) - self.var[winner, :]
        self.var[winner, :] += self.VARRATE * vdiff
        self.starv *= (1.0 - self.STARVRATE)
        self.starv[winner] += self.STARVRATE

    def produce_belief(self, sqdiff):
        """
        Update belief state.
        """
        sqdiff = np.asarray(sqdiff)
        normdist = theanoMatSum(theanoMatMatDiv(sqdiff, self.var), 1)
        # MatSum()
        chk = (normdist == 0)
        if any(chk):
            self.belief = np.zeros((1, self.CENTS))
            self.belief[chk] = 1.0
        else:
            normdist = np.array(theanoScaVecDiv(1, normdist))
            self.belief = np.array(theanoVecScaDiv(
                normdist, np.array(theanoVecSum(normdist)))).reshape(1, self.CENTS)

    def init_whitening(self, mn=[], st=[], tr=[]):
        """
        Initialize whitening parameters.
        """

        # Rescale initial means, since inputs will no longer
        # be non-negative.
        self.mean[:, self.LABDIMS:self.LABDIMS + self.EXTDIMS] *= 2.0
        self.mean[:, self.LABDIMS:self.LABDIMS + self.EXTDIMS] -= 1.0

        self.whitening = True
        self.patch_mean = mn
        self.patch_std = st
        self.whiten_mat = tr

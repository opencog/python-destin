# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# Numpy implementation of clustering
import numpy as np
import scipy.spatial.distance as npdist
from scipy import weave
from scipy.weave import converters


class Node:
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

        self.mean = np.random.rand(self.CENTS, self.DIMS)
        self.var = 0.001 * np.ones((self.CENTS, self.DIMS))
        self.starv = np.ones((self.CENTS, 1))
        self.belief = np.zeros((1, self.CENTS))

        self.children = []
        self.last = np.zeros((1, self.DIMS))
        self.whitening = False

    def update_node(self, input, TRAIN):
        """
        Update the node based on an input and training flag.
        """

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

        euc = np.sqrt(np.sum(sqdiff, axis=1)).reshape(self.CENTS, 1)
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
        vdiff = np.square(diff[winner, :]) - self.var[winner, :]  # this should be updated to use sqdiff
        self.var[winner, :] += self.VARRATE * vdiff
        self.starv *= (1.0 - self.STARVRATE)
        self.starv[winner] += self.STARVRATE

    def produce_belief(self, sqdiff):
        """
        Update belief state.
        """

        normdist = sum(sqdiff / self.var, axis=1)
        chk = (normdist == 0)
        if any(chk):
            self.belief = np.zeros((1, self.CENTS))
            self.belief[chk] = 1.0
        else:
            normdist = 1 / normdist
            self.belief = (normdist / sum(normdist)).reshape(1, self.CENTS)

    def add_child(self, child):
        """
        Add child node for providing input.
        """

        self.children.append(child)

    def latched_update(self, TRAIN):
        """
        Update node that has children nodes.
        """

        input = np.concatenate([c.belief for c in self.children], axis=1)
        self.update_node(input, TRAIN)

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

    def clear_belief(self):
        """
        Do nothing function so calling functions can be written
        to be compatible with both recurrent and non-recurrent
        clustering.
        """

        pass


class SupNode(Node):
    """
    This subclass implements supervised clustering. The first
    N dimensions of the input should be the label. Currently 
    a subclass of Node (e.g. no recurrent clustering).
    """

    def __init__(self, mr, vr, sr, di, ce, node_id, label_size):
        """
        Initialization function.
        """

        di += label_size
        # self.rec_init(mr, vr, sr, di, ce, node_id)
        self.common_init(mr, vr, sr, di, ce, node_id)
        self.SUP_MASK = np.zeros(self.DIMS, dtype=bool)
        self.SUP_MASK[label_size:] = True
        self.EXTDIMS = di - label_size
        self.LABDIMS = label_size

    def produce_belief(self, sqdiff):
        """
        Belief state update function.
        """

        normdist = np.sum(sqdiff[:, self.SUP_MASK] / self.var[:, self.SUP_MASK], axis=1)
        chk = (normdist == 0)
        if any(chk):
            self.belief = np.zeros((1, self.CENTS))
            self.belief[chk] = 1.0
        else:
            normdist = 1 / normdist
            self.belief = (normdist / sum(normdist)).reshape(1, self.CENTS)


    def update_node(self, input, TRAIN, label):
        """
        Update node based on input, label, and training flag.
        """

        input = input.reshape(1, self.EXTDIMS)
        if self.whitening and not (self.children):
            input = (input - self.patch_mean) / self.patch_std
            input = input.dot(self.whiten_mat)
        # input = concatenate([label.reshape(1,self.LABDIMS), input, self.belief], axis=1)
        input = np.concatenate([label.reshape(1, self.LABDIMS), input], axis=1)

        self.process_input(input, TRAIN)

    def latched_update(self, TRAIN, label):
        """
        Update node with children nodes.
        """

        input = np.concatenate([c.belief for c in self.children], axis=1)
        self.update_node(input, TRAIN, label)


def main():
    """
    Simple test function for development.
    """

    DIMS = 16
    CENTS = 30
    NEXAMPLES = 100000
    n1 = SupNode(0.01, 0.01, 0.001, DIMS, CENTS, 1, 5)
    input_array = np.random.rand(NEXAMPLES, DIMS)
    labels = np.zeros(NEXAMPLES * 5)
    tmp = np.random.randint(5, size=NEXAMPLES) + np.arange(NEXAMPLES) * 5
    labels[tmp] = 1.0
    labels.shape = (NEXAMPLES, 5)
    for row, label in zip(input_array, labels):
        n1.update_node(row, True, label)

    print n1.starv


if __name__ == "__main__":
    main()


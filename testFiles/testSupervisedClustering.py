__author__ = 'teddy'
from SupNode import *

DIMS = 16
CENTS = 30
NEXAMPLES = 100
mr = 0.01
vr = 0.01
sr = 0.001
node_id = 1
label_size = 5
sampleNode = Node(mr, vr, sr, DIMS, CENTS, node_id)
print sampleNode.starv
input_array = np.random.rand(NEXAMPLES, DIMS)
print("input dimensions")
print input_array.shape

labels = np.zeros(NEXAMPLES * 5)
tmp = np.random.randint(5, size=NEXAMPLES) + np.arange(NEXAMPLES) * 5
labels[tmp] = 1.0
labels.shape = (NEXAMPLES, 5)
for row in input_array:
    sampleNode.update_node(row, True)
print sampleNode.starv

"""
        self.LAYERS = 3
        self.CENTS_BY_LAYER = [25, 18, 25]
        self.MEAN_RATE = 0.01
        self.VAR_RATE = 0.01
        self.STARV_RATE = 0.001
        self.WINDOW_WIDTH = 4
        self.VIEW_WIDTH = self.WINDOW_WIDTH * pow(2, self.LAYERS-1)
        self.INPUT_SIZE = self.VIEW_WIDTH * self.VIEW_WIDTH
        self.IMAGE_WIDTH = image_size
"""
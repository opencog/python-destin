import numpy as np

Nx = 0
Ny = 0
Ratio = 2
# Input = np.random.rand((32,32))
for I in range(0, 32, Ratio):
    Ny = 0
    for J in range(0, 32, Ratio):
        print "%d %d" % (I, J)
        Ny += 1
    Nx += 1
print(Nx)
print(Ny)

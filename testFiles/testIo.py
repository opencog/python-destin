__author__ = 'teddy'
import scipy.io as io

myDict = {}
myDict['var1'] = 1
myDict['var2'] = 1
io.savemat('test.mat', myDict, appendmat=True)
myDict['var1'] = 2
myDict['var2'] = 2
io.savemat('test.mat', myDict, appendmat=True)
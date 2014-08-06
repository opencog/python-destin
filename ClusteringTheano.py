__author__ = 'teddy'
# Steven's incremental clustering re-implemented in Theano
import theano
import theano.tensor as T
from time import time
import math
import numpy as np
# implementing sample Theano functions

'''
def logFunctionOne(var1, var2):
    x = T.scalar('x')
    y = T.scalar('y')
    z = T.log10(x+y)
    f = theano.function([x,y],z)
    return f(var1, var2)
'''

def logMath(var1, var2):
    return math.log10(var1+var2)

def logTheano(var1, var2):
    return f(var1,var2)

A = [[10,10],[100,100]]
x = T.scalar('x')
y = T.scalar('y')
z = T.log10(x+y)
f = theano.function([x,y],z)
print('Theano')
start = time()
#print start
for I in range(1000):
    ans1 = logTheano(I,1)
    #print ans
elapsed1 = time() - start
print elapsed1
print('Math Module')
start = time()
for I in range(1000):
    ans2 = logMath(I,1)
    #print ans
elapsed2 = time() - start
print elapsed2
print('theano w.o function call')
start = time()
for I in range(1000):
    ans1 = f(I,1)
elapsed3 = time() - start
print elapsed3
#print('Diff %f' % (elapsed1-elapsed2))
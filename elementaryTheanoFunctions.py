__author__ = 'teddy'
from theano import function
import theano.tensor as T
import numpy as np
#  Implements basic operatioons needed by Learning Algorithms
#  Matrix-Matrix Addition, Division, Subtraction, Multiplication
#  Mat-Mat


def theanoMatMatAdd(In1, In2):
    var1 = T.dmatrix('var1')
    var2 = T.dmatrix('var2')
    var3 = T.add(var1, var2)
    AddMat = function([var1, var2], var3)
    return AddMat(In1, In2)


def theanoMatMatDiv(In1, In2):
    var1 = T.dmatrix('var1')
    var2 = T.dmatrix('var2')
    var3 = T.div_proxy(var1, var2)
    DivVec = function([var1, var2], var3)
    return DivVec(In1, In2)


def theanoMatMatSub(In1, In2):
    var1 = T.dmatrix('var1')
    var2 = T.dmatrix('var2')
    var3 = T.sub(var1, var2)
    SubMat = function([var1, var2], var3)
    return SubMat(In1, In2)


def theanoMatMatMul(In1, In2, option):
    if option == 'M':
        var1 = T.dmatrix('var1')
        var2 = T.dmatrix('var2')
        var3 = T.dot(var1, var2)
        MulMat = function([var1, var2], var3)
    else:
        var1 = T.dmatrix('var1')
        var2 = T.dmatrix('var2')
        var3 = T.mul(var1, var2)
        MulMat = function([var1, var2], var3)
    return MulMat(In1, In2)


def theanoMatSum(In1, axs):
    var1 = T.dmatrix('var1')
    MatSum = function([var1], T.sum(var1, axis=axs))
    return MatSum(In1)
#  Vec-Vec


def theanoVecScaDiv(In1, In2):
    var1 = T.dvector('var1')
    var2 = T.dscalar('var2')
    var3 = T.div_proxy(var1, var2)
    DivVec = function([var1, var2], var3)
    return DivVec(In1, In2)


def theanoScaVecDiv(In1, In2):
    var2 = T.dvector('var2')
    var1 = T.dscalar('var1')
    var3 = T.div_proxy(var1, var2)
    DivVec = function([var1, var2], var3)
    return DivVec(In1, In2)


def theanoVecVecAdd(In1, In2):
    var1 = T.dvector('var1')
    var2 = T.dvector('var2')
    var3 = T.add(var1, var2)
    AddVec = function([var1, var2], var3)
    return AddVec(In1, In2)


def theanoVecVecDiv(In1, In2):
    var1 = T.dvector('var1')
    var2 = T.dvector('var2')
    var3 = T.div_proxy(var1, var2)
    DivVec = function([var1, var2], var3)
    return DivVec(In1, In2)


def theanoVecVecSub(In1, In2):
    var1 = T.dvector('var1')
    var2 = T.dvector('var2')
    var3 = T.sub(var1, var2)
    DivVec = function([var1, var2], var3)
    return DivVec(In1, In2)


def theanoVecVecMul(In1, In2, opt):
    var1 = T.dvector('var1')
    var2 = T.dvector('var2')
    if opt == 'M':
        var3 = T.dot(var1, var2)
    else:
        var3 = T.mul(var1, var2)
    DivVec = function([var1, var2], var3)
    return DivVec(In1, In2)


def theanoVecSum(In1, axs=0):
    var1 = T.dvector('var1')
    var3 = T.sum(var1, axis=axs)
    VecSum = function([var1], var3)
    return VecSum(In1)
#  Mat-Sca


def theanoMatScaDiv(In1, In2):
    var1 = T.dmatrix('var1')
    var2 = T.dscalar('var2')
    var3 = T.div_proxy(var1, var2)
    Div = function([var1, var2], var3)
    return Div(In1, In2)


def theanoScaMatMul(In1, In2):
    var2 = T.dmatrix('var2')
    var1 = T.dscalar('var1')
    var3 = T.mul(var1, var2)
    Mul = function([var1, var2], var3)
    return Mul(In1, In2)


def theanoMatVecDiv(In1, In2):
    var1 = T.dmatrix('var1')
    var2 = T.dmatrix('var2')
    var3 = T.div_proxy(var1, var2)
    MVecDiv = function([var1, var2], var3)
    return MVecDiv(In1, In2)
# A = np.array([[1,2,3],[2,2,2]])
# B = np.array([[2,2,2],[1,1,1]])
#  print theanoMatMatAdd(A,B)

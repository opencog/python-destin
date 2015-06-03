"""
@author: Tejas Khot
@contact: tjskhot@gmail.com

@note: An implementation of a general Neural Network Layer
"""

import theano
import theano.tensor as T

import general_destin.nnfuns as nnf 
import general_destin.utils as utils


class Layer(object):
  """
  This class is an abstract layer for neural networks.
  
  A layer has an input set of neurons, and
  a hidden activation. The activation, f, is a
  function applied to the affine transformation
  of x by the connection matrix W, and the bias
  vector b.
  
  > y = f( W * x + b )
  """
  
  def __init__(self,
               input_size,
               hidden_size,
               is_recursive=False,
               activation_mode="tanh",
               weight_type="none",
               clip_gradients=False,
               clip_bound=1):
    """
    @param input_size: dimension of input data (int)
    @param hidden_size: dimension of hidden unit (int)
    @param is_recursive: True - RNN, False - Feedforward network (bool) 
    @param activation_mode: activation mode,
                          "tanh" for tanh function
                          "relu" for ReLU function
                          "sigmoid" for Sigmoid function
                          "softplus" for Softplus function
                          "linear" for linear function (string)
    @param clip_gradients: if use clip gradients to control weights (bool)
    @param clip_bound: this prevents explosion of gradients during backpropagation (int)
    """
    
    self.input_size=input_size
    self.hidden_size=hidden_size
    self.activation_mode=activation_mode
    
    # configure activation
    if (self.activation_mode=="tanh"):
     	self.activation=nnf.tanh
    elif (self.activation_mode=="relu"):
      	self.activation=nnf.relu
    elif (self.activation_mode=="sigmoid"):
      	self.activation=nnf.sigmoid
    elif (self.activation_mode=="softplus"):
      	self.activation=nnf.softplus
    elif (self.activation_mode=="softmax"):
      	self.activation=nnf.softmax
    else:
      	raise ValueError("Value %s is not a valid choice of activation function"
                       	% self.activation_mode)
                       
    self.clip_gradients=clip_gradients
    self.clip_bound=clip_bound
    self.is_recursive=is_recursive
    
    # initialize weights
    self.weight_type=weight_type
    self.get_weights(self.weight_type)
    
  def get_weights(self,
                  weight_type="none"):
    """
    Get network weights and bias
    Returns W,b
    
    @param weight_type: "none", "sigmoid", "tanh"
    
    @return: layer weights and bias
    """
    
    self.W=utils.get_shared_matrix("W", self.hidden_size, self.input_size, weight_type=weight_type)
    self.b=utils.get_shared_matrix("b", self.hidden_size, weight_type=weight_type)
    
  def get_pre_activation(self,
                         X):
    """
    Get pre-activation of the data
    Returns (W*X + b)
    
    @param data: assume data is row-wise
    
    @return: a pre-activation matrix
    """
    
    if self.clip_gradients is True:
      	X=theano.gradient.grad_clip(X, -self.clip_bound, self.clip_bound)
    
    return T.dot(X, self.W)+self.b
  
  def get_activation(self,
                     pre_activation):
    """
    Get activation from pre-activation
    i.e. f(W*X + b)
    
    @param pre_activation: pre-activation matrix
    
    @return: layer activation
    """
    
    return self.activation(pre_activation)
    
  def get_output(self,
                 X):
    """
    Get layer activation from input data
    
    @param X: input data, assume row-wise
    
    @return layer activation
    """
    
    return self.get_activation(self.get_pre_activation(X))
  
  @property
  def params(self):
    return (self.W, self.b)
  
  @params.setter
  def params(self, param_list):
    self.W.set_value(param_list[0].get_value())
    self.b.set_value(param_list[1].get_value())
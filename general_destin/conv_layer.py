"""
@author: Tejas Khot
@contact: tjskhot@gmail.com

@note: An implementation of a general Convolutional Neural Network Layer
"""
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

import general_destin.nnfuns as nnf 

class Layer(object):
	"""
	"""
	def __init__(self,
				 rng,
				 layer_num,
				 number_of_nodes,
				 feature_maps,
				 feature_shape,
				 filter_shape,
				 pool=False,
				 pool_size=(2,2),
				 stride_size=None,
				 border_mode="valid",
				 activate_mode="tanh"):
		"""
		Initialise the ConvNet Layer

		@param rng: random number generator for initializing weights (numpy.random.RandomState)
		@param layer_num: layer number in the Network 
		@param number_of_nodes: number of nodes in each layer (tuple or list of length 2)
		@param feature_maps: symbolic tensor of shape feature_shape (theano.tensor.dtensor4)
		@param feature_shape: (batch size, number of input feature maps,
		                       image height, image width) (tuple or list of length 4)
		@param filter_shape: (number of filters, number of input feature maps,
		                      filter height, filter width) (tuple or list of length 4)
		@param pool: Indicates if there is a max-pooling process in the layer. (bool)
		@param pool_size: the pooling factor (#rows, #cols) (tuple or list of length 2)
		@param stride_size: the number of shifts over rows/cols to get the the next pool region. 
		                    if st is None, it is considered equal to ds (no overlap on pooling regions) (tuple of size 2) 
		@param border_mode: convolution mode
		                    "valid" for valid convolution
		                    "full" for full convolution (string)  
		@param activate_mode: activation mode,
		                      "tanh" for tanh function
		                      "relu" for ReLU function
		                      "sigmoid" for Sigmoid function
		                      "softplus" for Softplus function
		                      "linear" for linear function (string)
		"""
		self.rng=rng
		self.layer_number=layer_num
        self.number_of_nodes=number_of_nodes 
		self.in_feature_maps=feature_maps
		self.feature_shape=feature_shape
		self.filter_shape=filter_shape
		self.pool=pool
		self.pool_size=pool_size
		self.stride_size=stride_size
		self.border_mode=border_mode
		self.activate_mode=activate_mode
		
		if (self.activate_mode=="tanh"):
		  self.activation=nnf.tanh
		elif (self.activate_mode=="relu"):
		  self.activation=nnf.relu
		elif (self.activate_mode=="sigmoid"):
		  self.activation=nnf.sigmoid
		elif (self.activate_mode=="softplus"):
		  self.activation=nnf.softplus
		elif (self.activate_mode=="softmax"):
		  self.activation=nnf.softmax
		elif (self.activate_mode=="linear"):
		  self.activation=nnf.linear
		else:
		  raise ValueError("Value %s is not a valid choice of activation function"
		                   % self.activate_mode)
		# generate weights and bias
		# There are (number of input feature maps * filter height * filter width) inputs to each hidden layer
		fan_in = np.prod(filter_shape[1:])
		# each unit in the lower layer receives a gradient from:
		# (number of output feature maps * filter height * filter width) / pooling_size
		fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(pool_size))
		    
		filter_bound = np.sqrt(6. / (fan_in + fan_out))
		self.filters = theano.shared(np.asarray(rng.uniform(low=-filter_bound,
		                                                    high=filter_bound,
		                                                    size=filter_shape),
		                                        			dtype='theano.config.floatX'),
		                             						borrow=True)
		self.filter_bound=filter_bound

		## Generate bias values, initialize as 0
		# the bias is a 1D tensor -- one bias per output feature map
		b_values = np.zeros((filter_shape[0],), dtype='theano.config.floatX')
		self.b = theano.shared(value=b_values, borrow=True)
		
		self.pooled, self.pooled_out=self.get_conv_pool(feature_maps=self.in_feature_maps,
		                                                feature_shape=self.feature_shape,
		                                                filters=self.filters,
		                                                filter_shape=self.filter_shape,
		                                                bias=self.b,
		                                                pool=self.pool,
		                                                pool_size=self.pool_size,
		                                                border_mode=self.border_mode)
		                                   
		self.out_feature_maps=self.get_activation(self.pooled_out)
		
		self.params=[self.filters, self.b]
		  
		def get_conv_pool(self,
		                  feature_maps,
		                  feature_shape,
		                  filters,
		                  filter_shape,
		                  bias,
		                  subsample=(1,1),
		                  pool=False,
		                  pool_size=(2,2),
		                  border_mode="valid"):
		  
		  """
		  Convolution operation in this version is not as powerful as using dnn_conv
		  """
		  
		  conv_out=conv.conv2d(input=feature_maps,
		                       filters=filters, 
		                       image_shape=feature_shape,
		                       filter_shape=filter_shape,
		                       border_mode=border_mode,
		                       subsample=subsample)
		  
		  if pool==True:
		    pooled_out=downsample.max_pool_2d(input=conv_out,
		                                      ds=pool_size)
		  else:
		    pooled_out=conv_out
		    
		  return pooled_out, pooled_out+bias.dimshuffle("x", 0, "x", "x")
		
		def get_activation(self,
		                   pooled_out):
		  return self.activation(pooled_out)
		
		def get_out_feature_maps(self):
		  return self.out_feature_maps
		
		def get_output(self, x):
		  
		  _, pooled_out=self.get_conv_pool(feature_maps=x,
		                                   feature_shape=self.feature_shape,
		                                   filters=self.filters,
		                                   filter_shape=self.filter_shape,
		                                   bias=self.b,
		                                   pool=self.pool,
		                                   pool_size=self.pool_size,
		                                   border_mode=self.border_mode)
		  
		  return self.get_activation(pooled_out)





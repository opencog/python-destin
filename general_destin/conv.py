"""
@author: Tejas Khot
@contact: tjskhot@gmail.com

@note: An implementation of a general Convolutional Neural Network Layer

TODO:
Add cuda-convnet wrappers from pylearn2 for:
ProbMaxPool, StochasticMaxPool, WeightedMaxPool, CrossMapNorm
preferably in a separate pooling file
"""
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.pool import MaxPool

import general_destin.nnfuns as nnf 

class ConvLayer(object):
	"""
	This class implements a general layer of a Convolutional Neural Network
	"""
	def __init__(self,
		     rng,
		     input,
		     image_shape,
		     filter_shape,
		     pool=False,
		     pool_size=(2,2),
		     activation_mode="tanh",
		     stride_size=None,
		     border_mode="valid",
                     tied_biases=False):
		"""
		Initialise the ConvNet Layer

		@param rng: random number generator for initializing weights (numpy.random.RandomState)
		@param input: symbolic tensor of shape image_shape (theano.tensor.dtensor4)
		@param image_shape: (batch size, number of input feature maps,
		                       image height, image width) (tuple or list of length 4)
		@param filter_shape: (number of filters, number of input feature maps,
		                      filter height, filter width) (tuple or list of length 4)
		@param pool: Indicates if there is a max-pooling process in the layer. (bool)
		@param pool_size: the pooling factor (#rows, #cols) (tuple or list of length 2)
		@param activate_mode: activation mode,
	                              "tanh" for tanh function
	                              "relu" for ReLU function
	                              "sigmoid" for Sigmoid function
	                              "softplus" for Softplus function
	                              "linear" for linear function (string)
		@param stride_size: the number of shifts over rows/cols to get the the next pool region. 
		                    if st is None, it is considered equal to ds (no overlap on pooling regions) (tuple of size 2) 
		@param border_mode: convolution mode
		                    "valid" for valid convolution
		                    "full" for full convolution (string)  
	        @param tied_biases: one single bias per feature map
				    Note that untied biases train much faster.
				    The tradeoff is between tied biases leading to underfitting and untied biases leading to overfitting
		"""
		self.rng=rng
		self.layer_number=layer_num
		self.in_input=input
		self.image_shape=image_shape
		self.filter_shape=filter_shape
		self.pool=pool
		self.pool_size=pool_size
		self.stride_size=stride_size
		self.border_mode=border_mode
		self.tied_biases=tied_biases
	
	def initialize(self):
		# generate weights and bias
		# There are (number of input feature maps * filter height * filter width) inputs to each hidden layer
		fan_in=np.prod(filter_shape[1:])
		# each unit in the lower layer receives a gradient from:
		# (number of output feature maps * filter height * filter width) / pooling_size
		fan_out=(filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(pool_size))
		    
		W_bound=np.sqrt(6. / (fan_in + fan_out))
		self.W=theano.shared(np.asarray(self.rng.uniform(low=-W_bound,
	                                                      	   high=W_bound,
	                                                      	   size=filter_shape),
                                		         	   dtype='theano.config.floatX'),
                     						   borrow=True)

		## Generate bias values, initialize as 0
		# the bias is a 1D tensor -- one bias per output feature map
		b_values=np.zeros((filter_shape[0],), dtype='theano.config.floatX')
		self.b=theano.shared(value=b_values, borrow=True)
		
		self.pooled_out=self.apply_conv(input=self.in_input,
	                                        image_shape=self.image_shape,
	                                        filters=self.W,
	                                        filter_shape=self.filter_shape,
	                                        bias=self.b,
	                                        pool=self.pool,
	                                        pool_size=self.pool_size,
	                                        border_mode=self.border_mode)
                               
		self.out_feature_maps=self.get_activation(self.pooled_out)
		
	
	def apply_conv(	self,
			input,
			image_shape,
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
	    	output=conv.conv2d(input=input,
	             		   filters=filters, 
	                  	   image_shape=image_shape,
	                   	   filter_shape=filter_shape,
	                           border_mode=border_mode,
	                           subsample=subsample)
	    	
	    	if self.tied_biases:
	      	  	pooled_out+=bias.dimshuffle("x", 0, "x", "x")
	      	else:
	      		pooled_out+=bias.dimshuffle('x', 0, 1, 2)
	      	"""
	      	TODO : Replace this pool part with separate classes that offers other kinds of pooling
	      	"""
	        if pool==True:
	         	pooled_out=downsample.max_pool_2d(input=output, ds=pool_size)
	    	else:
	      		pooled_out=output      		
	      	return pooled_out   
      
      
	def get_activation(self, pooled_out):
		return self.activation(pooled_out)
		 
	def get_out_feature_maps(self):
	  	return self.out_feature_maps
	
	def get_output(self, x):	  
		pooled_out=self.apply_conv( input=x,
		                       	    image_shape=self.image_shape,
		                            filters=self.W,
		                            filter_shape=self.filter_shape,
		                            bias=self.b,
		                            pool=self.pool,
		                            pool_size=self.pool_size,
		                            border_mode=self.border_mode)
	  
	  	return self.get_activation(pooled_out)
	def get_dim(self, name):
		"""""
		Get dimensions of an input/output variable
		
		@param name: name of the variable (str)
		
		@return dimensions of the variable
		"""
        	raise ValueError("No dimension information for {} available"
				.format(name))

	@property
	def params(self):
		return (self.W, self.b)
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
			     image_shape,
			     filter_shape,
			     pool=False,
			     pool_size=(2,2),
			     activation_mode="tanh",
			     stride=(1, 1),
			     border_mode="valid",
                 tied_biases=False):
		"""
		Initialise the ConvNet Layer

		@param rng: random number generator for initializing weights (numpy.random.RandomState)
		@param input: symbolic tensor of shape image_shape (theano.tensor.dtensor4)
		@param image_shape: (batch size, number of input feature maps,image height, image width) (tuple or list of length 4)
		@param filter_shape: (number of filters, number of input feature maps, filter height, filter width) (tuple or list of length 4)
		@param pool: indicates if there is a max-pooling process in the layer. Defaults to False. (bool)
		@param pool_size: the pooling factor (#rows, #cols) (tuple or list of length 2)
		@param activation_mode: activation mode,
								"tanh" for tanh function
								"relu" for ReLU function
								"sigmoid" for Sigmoid function
								"softplus" for Softplus function
								"linear" for linear function (string)
		@param stride: 	the number of shifts over rows/cols to get the the next pool region. 
	                    if st is None, it is considered equal to ds (no overlap on pooling regions) (tuple of size 2) 
	                    The step (or stride) with which to slide the filters over the image. Defaults to (1, 1).
		@param border_mode: convolution mode
		                    "valid" for valid convolution
		                    "full" for full convolution (string)  
        @param tied_biases: one single bias per feature map (bool)
						    Note that untied biases train much faster.
						    The tradeoff is between tied biases leading to underfitting and untied biases leading to overfitting
		"""
		self.rng=rng
		self.layer_number=layer_num
		self.input=input
		self.image_shape=image_shape
		self.filter_shape=filter_shape
		self.pool=pool
		self.pool_size=pool_size
		self.stride=stride
		self.border_mode=border_mode
		self.tied_biases=tied_biases
		
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
	
	def initialize(self):
		"""
		Set values for weights and biases
		"""
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
		# Generate bias values, initialize as 0
		# the bias is a 1D tensor -- one bias per output feature map when using untied biases
		b_values=np.zeros((filter_shape[0],), dtype='theano.config.floatX')
		self.b=theano.shared(value=b_values, borrow=True)
		# These two would be assigned only when called
		self.output=None
		self.feature_maps=None
				
	
	def apply_conv(self, input):
		"""
		This method applies the convolution operation on the input provided
	
    	@note Convolution operation in this version is not as powerful as using dnn_conv
    	
    	@param input: symbolic tensor of shape image_shape (theano.tensor.dtensor4)
    			      A 4D tensor with the axes representing batch size, number of
       			      channels, image height, and image width.
     	----------------------------------------------------------------------------------
		@return output : A 4D tensor of filtered images (feature maps) with dimensions
			            representing batch size, number of filters, feature map height,
			            and feature map width.
			
			            The height and width of the feature map depend on the border
			            mode. For 'valid' it is ``image_size - filter_size + 1`` while
			            for 'full' it is ``image_size + filter_size - 1``
    	"""
    	self.output=conv.conv2d(input=self.input,
	             		   	    filters=self.W, 
	                  	   	    image_shape=self.image_shape,
	                   	   	    filter_shape=self.filter_shape,
	                            border_mode=self.border_mode,
	                            subsample=self.stride)	    	
    	if self.tied_biases:
      	  	self.output+=self.b.dimshuffle("x", 0, "x", "x")
      	else:
      		self.output+=self.b.dimshuffle('x', 0, 1, 2)
      	"""
      	TODO : Replace this pool part with separate classes that offers other kinds of pooling
      	"""
        if pool==True:
         	self.output=downsample.max_pool_2d(input=output, ds=pool_size)
      	return self.output   
      
	def apply_activation(self, pre_activation):
	    """
	    Apply activation on pre-activation input
	    i.e. f(pre_activation)
	    
	    @param pre_activation: pre-activation matrix
	    ---------------------------------------------
	    @return: layer activation
	    """
	    self.feature_maps=self.activation(pre_activation)
	    return self.feature_maps
		 
	def get_feature_maps(self):
	  	return self.feature_maps
	
	def get_output(self, x):
		return self.output

	def get_dim(self, name):
		"""""
		Get dimensions of an input/output variable
		
		@param name: name of the variable (str)
	 	------------------------------------------------
		@return dimensions of the variable
		"""
    	raise ValueError("No dimension information for {} available"
						.format(name))

	@property
	def params(self):
		return (self.W, self.b)
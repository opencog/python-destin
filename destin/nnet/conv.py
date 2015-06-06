"""
@author: Tejas Khot
@contact: tjskhot@gmail.com

@note: An implementation of a general Convolutional Neural Network Layer

TODO:
Add cuda-convnet wrappers from pylearn2 for:
ProbMaxPool, StochasticMaxPool, WeightedMaxPool, CrossMapNorm
preferably in a separate pooling file
"""
from pipes import stepkinds

import nnfuns as nnf 
import numpy as np
import theano
from theano.sandbox.cuda.basic_ops import gpu_contiguous
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal.downsample import max_pool_2d


try:
    from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
    from pylearn2.sandbox.cuda_convnet.pool import MaxPool
except ImportError:
    print "Note: pylearn2 not available, FilterActs cannot be used"
    pylearn2=None


class ConvLayer(object):
    """
    This class implements a general layer of a Convolutional Neural Network
    """
    def __init__(self,
                 image_shape,
                 filter_shape,
                 pool=False,
                 pool_size=(1,1),
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
        self.image_shape=image_shape
        self.filter_shape=filter_shape
        self.pool=pool
        self.pool_size=pool_size
        self.stride=stride
        self.border_mode=border_mode
        self.tied_biases=tied_biases
        #randomly chosen seed
        self.rng=np.random.RandomState(23455)
		
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
		
        @note 
        Weights are sampled randomly from a uniform distribution in the range [-1/fan-in, 1/fan-in], 
        where fan-in is the number of inputs to a hidden unit.
        """
        # generate weights and bias
        # There are (number of input feature maps * filter height * filter width) inputs to each hidden layer
        fan_in=np.prod(self.filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # (number of output feature maps * filter height * filter width) / pooling_size
        fan_out=(self.filter_shape[0] * np.prod(self.filter_shape[2:]) / np.prod(self.pool_size))
        
        W_bound=np.sqrt(6. / (fan_in + fan_out))
        self.W=theano.shared(np.asarray(self.rng.uniform(low=-W_bound,
                                                         high=W_bound,
                                                         size=self.filter_shape),
                                                         type='theano.config.floatX'),
                                                         borrow=True)
        # Generate bias values, initialize as 0
        # the bias is a 1D tensor -- one bias per output feature map when using untied biases
        b_values=np.zeros((self.filter_shape[0],), dtype='theano.config.floatX')
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
        self.output=conv.conv2d(input=input,
                                filters=self.W, 
                                image_shape=self.image_shape,
                                filter_shape=self.filter_shape,
                                border_mode=self.border_mode,
                                subsample=self.stride)	    	
        if self.tied_biases:
            self.output+=self.b.dimshuffle("x", 0, "x", "x")
        else:
            self.output+=self.b.dimshuffle('x', 0, 1, 2)
      		
        if pool==True:
            self.pooling=MaxPooling(self.pool_size, self.stride)
            self.output=self.pooling.apply(self.output)
         	
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

    @property
    def params(self):
        return (self.W, self.b)


class ConvLayerFilterActs(ConvLayer):
    """
    This class implements a general layer of ConvNet using FilterActs wrapper from pylearn2 sandbox
    """
    def __init__(self,
                 image_shape,
                 filter_shape,
                 pool=False,
                 pool_size=(1,1),
                 activation_mode="tanh",
                 stride=(1, 1),
                 border_mode="valid",
                 tied_biases=False):
        """
        Initialise the ConvNet Layer
        
        @note: Supports only max pooling for now
        
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
        if pylearn2 is None:
            raise ImportError("Note: pylearn2 not available, FilterActs cannot be used")
        super(ConvLayerFilterActs, self).__init__(image_shape, filter_shape, pool, pool_size, 
                                                  activation_mode, stride, border_mode, tied_biases)
	
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
        ----------------------------------------------------------------------------------
        Limitations of using FilterActs compared to conv2d:
        
        > Number of channels <= 3; If you want to compute the gradient, it should be divisible by 4.
        > Filters must be square.
        > Number of filters must be a multiple of 16
        > All minibatch sizes are supported, but the best performance is achieved when the minibatch size 
        is a multiple of 128.
        > Works only on the GPU
        """
        input_shuffled=input.dimshuffle(1, 2, 3, 0) # bc01 to c01b
        filters_shuffled=self.W.dimshuffle(1, 2, 3, 0) # bc01 to c01b
        ## Use zero padding with (filter_size - 1) border i.e. full convolution
        if self.border_mode=="full":
            padding=self.filter_shape[0]-1
        else:
            padding=0
        conv_out=FilterActs(stride=1, partial_sum=1, pad=padding)
        contiguous_input=gpu_contiguous(input_shuffled)
        contiguous_filters=gpu_contiguous(filters_shuffled)
        conv_out_shuffled=conv_out(contiguous_input, contiguous_filters)
        if pool==True:
            pool_op=MaxPool(ds=pool_size[0], stride=pool_size[0])
            pooled_out_shuffled=pool_op(conv_out_shuffled)
            pooled_out=pooled_out_shuffled.dimshuffle(3, 0, 1, 2) # c01b to bc01
            pooled_out=downsample.max_pool_2d(input=conv_out,
                                              ds=pool_size)
        else:
            pooled_out=conv_out
		
        self.output=pooled_out
        
        if self.tied_biases:
            self.output+=self.b.dimshuffle("x", 0, "x", "x")
        else:
            self.output+=self.b.dimshuffle('x', 0, 1, 2)
            
        return self.output


class MaxPooling(object):
    """
    This class performs Max Pooling
    """
    def __init__(self,
                 pooling_size,
                 stride=None):
        """
        @param pooling_size: The height and width of the pooling region i.e. this is the factor
                             by which your input's last two dimensions will be downscaled.
        @param stride:  The vertical and horizontal shift (stride) between pooling regions.
                        By default this is (1,1) i.e. same as pool_size. 
                        Setting this to a lower value than pool_size results in overlapping pooling regions.
        """
        self.pooling_size=pooling_size
        self.stride=stride
        
    def apply(self, input):
        """
        Apply the pooling (subsampling) transformation.
        
        @param input: An tensor with dimension greater or equal to 2. The last two
                      dimensions will be downsampled. For example, with images this
                      means that the last two dimensions should represent the height
                      and width of your image.
        """
        pooled_output=max_pool_2d(input, self.pooling_size, st=self.stride)
        return pooled_output
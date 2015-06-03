"""
@author: Tejas Khot
@contact: tjskhot@gmail.com

@note: An implementation of some utility functions for Neural Networks
"""

import theano
import theano.tensor as T

def get_shared_matrix(name,
                      out_size,
                      in_size=None,
                      weight_type="none"):
  """
  create shared matrix or vector of weights using the given in_size and out_size
  
  @param out_size: output size (int)
  @param in_size: input size (int)
  @param weight_type: "none", "tanh" and "sigmoid"
  
  @return: a shared matrix with size of (in_size x out_size) initialized randomly
        OR a vector of size (out_size, )
  """
  
  if in_size==None:
    return theano.shared(value=np.asarray(np.random.uniform(low=0,
                                                            high=1./out_size,
                                                            size=(out_size, )),
                                                            dtype="float32"),
                                                            name=name,
                                                            borrow=True)
  else:
        if weight_type=="tanh":
            lower_bound=-np.sqrt(6. / (in_size + out_size))
            upper_bound=np.sqrt(6. / (in_size + out_size))
        elif weight_type=="sigmoid":
            lower_bound=-4*np.sqrt(6. / (in_size + out_size))
            upper_bound=4*np.sqrt(6. / (in_size + out_size))
        elif weight_type=="none":
            lower_bound=0
            upper_bound=1./(in_size+out_size)
        return theano.shared(value=np.asarray(np.random.uniform(low=lower_bound,
                                                                high=upper_bound,
                                                                size=(in_size, out_size)),
                                                                dtype="float32"),
                                                                name=name,
                                                                borrow=True)

def Dropout(shape, prob):
    pass

def MultiDropout(shape, dropout=0.):
    pass
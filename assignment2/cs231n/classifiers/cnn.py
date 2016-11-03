import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    stride = self.conv_param["stride"]
    pad = self.conv_param["pad"]
    pool_height = self.pool_param["pool_height"]
    pool_width = self.pool_param["pool_width"]
    pool_stride = self.pool_param["stride"]

    C, H, W = input_dim
    F, HH, WW = (num_filters, filter_size, filter_size)

    # convo layer size
    Hp = 1 + (H + 2 * pad - HH) / stride
    Wp = 1 + (W + 2 * pad - WW) / stride

    # pooling
    Hp_pool = 1 + (Hp - pool_height) / pool_stride
    Wp_pool = 1 + (Wp - pool_width) / pool_stride

    # first layer, convo
    self.params["W1"] = weight_scale * np.random.randn(F, C, HH, WW)
    self.params["b1"] = np.zeros(F)

    # hidden affine layer
    self.params["W2"] = weight_scale * np.random.randn(F*Hp_pool*Wp_pool, hidden_dim)
    self.params["b2"] = np.zeros(hidden_dim)

    # output affine layer
    self.params["W3"] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params["b3"] = np.zeros(num_classes)

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    out_conv, cache_conv = conv_relu_pool_forward(X, W1, b1, self.conv_param, self.pool_param)
    out_hidden, cache_hidden = affine_relu_forward(out_conv, W2, b2)
    scores, cache_output = affine_forward(out_hidden, W3, b3)

    if y is None:
      return scores
    
    loss, grads = 0, {}

    loss, F = softmax_loss(scores, y)
    dx3, dw3, db3 = affine_backward(F, cache_output)
    grads["W3"] = dw3
    grads["b3"] = db3

    dx2, dw2, db2 = affine_relu_backward(dx3, cache_hidden)
    grads["W2"] = dw2
    grads["b2"] = db2

    dx1, dw1, db1 = conv_relu_pool_backward(dx2, cache_conv)
    grads["W1"] = dw1
    grads["b1"] = db1

    for k, v in self.params.items():
      if "W" in k:
        grads[k] += self.reg * self.params[k]
        loss += 0.5 * self.reg * (np.sum(self.params[k] ** 2))
    
    return loss, grads

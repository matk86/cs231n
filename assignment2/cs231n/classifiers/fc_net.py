import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b2'] = np.zeros(num_classes)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """
    grads = {}
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']

    out1, cache1 = affine_forward(X, W1, b1)
    out2, cache2 = relu_forward(out1)
    scores, cache3 = affine_forward(out2, W2, b2)

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores

    loss, F = softmax_loss(scores, y)
    #print F.shape
    dx2, dw2, db2 = affine_backward(F, cache3)
    #print dx2.shape, dw2.shape, db2.shape
    dx2 = relu_backward(dx2.reshape(out2.shape), out2)
    #print dx2.shape
    dx1, dw1, db1 = affine_backward(dx2, cache1)

    dw1 += self.reg * W1
    dw2 += self.reg * W2
    loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2))

    grads["W1"] = dw1
    grads["W2"] = dw2
    grads["b1"] = db1
    grads["b2"] = db2

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    # first layer
    self.params["W1"] = weight_scale * np.random.randn(input_dim, hidden_dims[0])
    self.params["b1"] = np.zeros(hidden_dims[0])

    # hidden layers
    for l in range(1, self.num_layers-1):
      self.params["W"+str(int(l+1))] = weight_scale * np.random.randn(hidden_dims[l-1], hidden_dims[l])
      self.params["b"+str(int(l+1))] = np.zeros(hidden_dims[l])
            

    # last layer
    self.params["W"+str(int(self.num_layers))] = weight_scale * np.random.randn(hidden_dims[-1], num_classes)
    self.params["b"+str(int(self.num_layers))] = np.zeros(num_classes)

    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
      for l in range(0, self.num_layers-1):      
        self.params["gamma"+str(int(l+1))] = np.ones(hidden_dims[l])
        self.params["beta"+str(int(l+1))] = np.zeros(hidden_dims[l])    

    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    cache_dropout = None
    cache_bn = None

    # dropout
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode
      cache_dropout = []

    # BN
    if self.use_batchnorm:
      gamma = []
      beta = []
      for bn_param in self.bn_params:
        bn_param[mode] = mode
      for l in range(self.num_layers-1):      
        gamma.append(self.params["gamma"+str(int(l+1))])
        beta.append(self.params["beta"+str(int(l+1))])            
      cache_bn = []        

    W = []
    b = []
    out2_list = []
    cache = []
    for l in range(self.num_layers):
      W.append(self.params["W"+str(int(l+1))])
      b.append(self.params["b"+str(int(l+1))])

    # input layer
    out1, cache1 = affine_forward(X, W[0], b[0])
    if self.use_batchnorm:
        out1, cbn = batchnorm_forward(out1, gamma[0], beta[0], self.bn_params[0])
        cache_bn.append(cbn)      
    out2, _ = relu_forward(out1)
    if self.use_dropout:
      out2, cd = dropout_forward(out2, self.dropout_param)
      cache_dropout.append(cd)
    out2_list.extend([out2])
    cache.extend([cache1])

    i = 0
    # hidden layers
    for Wi, bi in zip(W[1:-1], b[1:-1]):
      out1, cache1 = affine_forward(out2, Wi, bi)
      if self.use_batchnorm:
        gammai, betai, bnp =  gamma[i+1], beta[i+1], self.bn_params[i+1]
        out1, cbn = batchnorm_forward(out1, gammai, betai, bnp)
        cache_bn.append(cbn)
        i += 1
      out2, _ = relu_forward(out1)
      if self.use_dropout:
        out2, cd = dropout_forward(out2, self.dropout_param)
        cache_dropout.append(cd)
      out2_list.extend([out2])  # num_layers - 1
      cache.extend([cache1])  # num_layers - 1

    # output layer
    scores, cache3 = affine_forward(out2, W[-1], b[-1])

    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}

    # output layer gradient
    loss, F = softmax_loss(scores, y)
    dx2, dw2, db2 = affine_backward(F, cache3)
    grads["W"+str(int(self.num_layers))] = dw2
    grads["b"+str(int(self.num_layers))] = db2

    
    for i in range(self.num_layers-1, 0, -1):
      if self.use_dropout:
        dx2 = dropout_backward(dx2, cache_dropout[i-1])
      dx2 = relu_backward(dx2.reshape(out2_list[i-1].shape), out2_list[i-1])
      if self.use_batchnorm:
        dx2, dgamma, dbeta = batchnorm_backward(dx2, cache_bn[i-1])
        grads["gamma" + str(int(i))] = dgamma
        grads["beta" + str(int(i))] = dbeta
      dx2, dw1, db1 = affine_backward(dx2, cache[i-1])
      grads["W" + str(int(i))] = dw1
      grads["b" + str(int(i))] = db1

    for k, v in self.params.items():
      if "W" in k:
        grads[k] += self.reg * self.params[k]
        loss += 0.5 * self.reg * (np.sum(self.params[k] ** 2))

    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #

    return loss, grads

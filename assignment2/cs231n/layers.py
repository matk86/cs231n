import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  xshape_orig = x.shape
  x = x.reshape((x.shape[0], -1))
  out = x.dot(w) + b
  x = x.reshape(xshape_orig)
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  xshape_orig = x.shape
  x = x.reshape((x.shape[0], -1)) # N, D: D = d1*d2..*dk
  #xw = x.dot(w) # N, M
  #H = np.where(xw>0, 1, 0) # N, M

  dH = dout #* H / xshape_orig[0] # N, M
  dw = (x.T).dot(dH)  # D, M
  db = np.sum(dH, axis=0)  # 1, M
  dx = dH.dot(w.T)  # N, D
  dx = dx.reshape(xshape_orig)  # N, d1, ..., d_k

  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = np.where(x>0, x, 0)
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  dx = np.where(x > 0, 1, 0) * dout
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  sample_mean = np.mean(x, axis=0)
  sample_var = np.var(x, axis=0)
  sample_std = np.sqrt(sample_var + eps)

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var
  running_std = np.sqrt(running_var + eps)
  
  if mode == 'train':
    mean = sample_mean
    std = sample_std
  elif mode == 'test':
    mean = running_mean
    std = running_std
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  xmean = x - mean
  xhat = xmean / std
  
  out = gamma * xhat + beta

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  cache = (xmean, xhat, mean, std, gamma, beta, bn_param)

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  xmean, xhat, sample_mean, sample_std, gamma, beta, bn_param = cache
  N = dout.shape[0]
  # wrt beta
  dbeta = np.sum(dout, axis=0)
  # wrt gamma
  dgamma = np.sum(dout * xhat, axis=0)
  # wrt xhat
  dxhat = dout * gamma
  if bn_param["mode"] == 'train':  
    # wrt variance
    dvar = -np.sum(dxhat * xmean / (sample_std**3), axis=0) / 2.
    # wrt x
    dx = dxhat/sample_std + 2*dvar*xmean / N
    dx -=  np.mean(dx, axis=0)
  elif bn_param["mode"] == 'test':
    dx = dxhat / sample_std

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the alt backward pass for batch normalization.
  # Store the results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  N,  D = x.shape
  mask = np.random.rand(N, D) < p
  out = None

  if mode == 'train':
    out = np.where(mask, x, 0)
  elif mode == 'test':
    out = x

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    dx = np.where(mask, dout, 0)
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  stride = conv_param["stride"]
  pad = conv_param["pad"]
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  Hp = 1 + (H + 2 * pad - HH) / stride
  Wp = 1 + (W + 2 * pad - WW) / stride

  xpadded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
  out = np.zeros((N, F, Hp, Wp))  # N, F, Hp, Wp

  for I in range(N):
    for si in range(Hp):
      for sj in range(Wp):
        i_begin = stride * si
        i_end = i_begin + HH

        j_begin = stride * sj
        j_end = j_begin + WW

        xblock = xpadded[I, :, i_begin:i_end, j_begin:j_end]
        xblock = xblock.reshape((-1, 1))  # C*HH*WW, 1
        w_f = w.reshape((F, -1))  # F, C*HH*WW
        out[I, :, si, sj] = w_f.dot(xblock).ravel() + b  # F, 1

        #xblock_array = xblock.ravel()  # 1, C*HH*WW
        #for f in range(F):
        #  w_array = w[f, :, :, :].ravel()
        #  out[I, f, si, sj] = xblock_array.dot(w_array) + b[f]

  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives. N, F, Hp, Wp
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x, (N, C, H, W)
  - dw: Gradient with respect to w, (F, C, HH, WW)
  - db: Gradient with respect to b, (F)
  """
  dx, dw, db = None, None, None
  x, w, b, conv_param = cache
  stride = conv_param["stride"]
  pad = conv_param["pad"]
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  Hp = 1 + (H + 2 * pad - HH) / stride
  Wp = 1 + (W + 2 * pad - WW) / stride

  xpadded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
  dw = np.zeros((F, C, HH, WW))  # F, C, HH, WW
  dxpadded = np.zeros_like(xpadded)  # N, C, H, W

  for I in range(N):
    for si in range(Hp):
      for sj in range(Wp):
        i_begin = stride * si
        i_end = i_begin + HH

        j_begin = stride * sj
        j_end = j_begin + WW

        xblock = xpadded[I, :, i_begin:i_end, j_begin:j_end]  # C, HH, WW

        for f in range(F):
          dw[f, :, :, :] += xblock * dout[I, f, si, sj]
          dxpadded[I, :, i_begin:i_end, j_begin:j_end] +=  w[f, :, :, :] * dout[I, f, si, sj]

  db = np.sum(dout, axis=(0, 2, 3))
  dx = dxpadded[:,:, pad:-pad, pad:-pad]
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  pool_height = pool_param["pool_height"]
  pool_width = pool_param["pool_width"]
  stride = pool_param["stride"]
  N, C, H, W = x.shape
  Hp = 1 + (H - pool_height) / stride
  Wp = 1 + (W - pool_width) / stride

  out = np.zeros((N, C, Hp, Wp))  # N, C, Hp, Wp

  for I in range(N):
    for si in range(Hp):
      for sj in range(Wp):
        i_begin = stride * si
        i_end = i_begin + pool_height

        j_begin = stride * sj
        j_end = j_begin + pool_width
        for c in range(C):
          xblock = x[I, c, i_begin:i_end, j_begin:j_end]
          out[I, c, si, sj] = np.max(xblock)
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height = pool_param["pool_height"]
  pool_width = pool_param["pool_width"]
  stride = pool_param["stride"]
  N, C, H, W = x.shape
  Hp = 1 + (H - pool_height) / stride
  Wp = 1 + (W - pool_width) / stride

  dx = np.zeros(x.shape)  # N, C, Hp, Wp

  for I in range(N):
    for si in range(Hp):
      for sj in range(Wp):
        i_begin = stride * si
        i_end = i_begin + pool_height

        j_begin = stride * sj
        j_end = j_begin + pool_width
        for c in range(C):
          xblock = x[I, c, i_begin:i_end, j_begin:j_end]
          max_arg = np.argwhere(abs(xblock - np.max(xblock)) < 1e-10)[0]
          i_max = max_arg[0] + i_begin
          j_max = max_arg[1] + j_begin
          dx[I, c, i_max, j_max] = dout[I, c, si, sj]
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.

  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, C, H, W = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(C, dtype=x.dtype))

  sample_mean = np.mean(x, axis=(0, 2, 3))
  sample_var = np.var(x, axis=(0, 2, 3))

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var
  if mode == 'train':
    xhat = x - sample_mean[np.newaxis, :, np.newaxis, np.newaxis]
    xhat /= np.sqrt(sample_var[np.newaxis, :, np.newaxis, np.newaxis] + eps)
  elif mode == 'test':
    xhat = x - running_mean[np.newaxis, :, np.newaxis, np.newaxis]
    xhat /= np.sqrt(running_var[np.newaxis, :, np.newaxis, np.newaxis] + eps)
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  out = gamma[np.newaxis, :, np.newaxis, np.newaxis] * x + beta[np.newaxis, :, np.newaxis, np.newaxis]

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  cache = (x, sample_mean, sample_var, eps, gamma, beta)

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.

  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  pass

  return dx, dgamma, dbeta


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

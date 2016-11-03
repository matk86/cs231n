import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    loss_score = 0.0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      loss_score += np.exp(scores[j])
    loss +=  np.log(loss_score) - correct_class_score
    for j in xrange(num_classes):
      if j == y[i]:
        dW[:, j] -= X[i]
      else:
        dW[:, j] += X[i]/loss_score * np.exp(scores[j])

  loss /= num_train

  loss += 0.5 * reg * np.sum(W*W)
  dW += reg * W
  dW /= num_train
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  N, D = X.shape
  _, C = W.shape

  XW = X.dot(W)

  correct_class_score = XW[np.arange(0, N), y].copy()
  loss_matrix = np.exp(XW)
  # normalize
  #loss_vec = np.sum(loss_matrix, axis=1) - np.exp(correct_class_score)
  #loss_matrix = loss_matrix / loss_vec[:, np.newaxis]
  loss_matrix[np.arange(0, N), y] = 0
  loss_vec = np.sum(loss_matrix, axis=1)
  loss = np.sum(np.log(loss_vec) - correct_class_score)
  loss /= N
  loss += 0.5 * reg * np.sum(W * W)

  F = np.zeros((N, C))
  # j!= y_i
  F = np.exp(XW)/ loss_vec[:, np.newaxis]
  # j = y_i
  F[np.arange(0, N), y] = -1 
  dW = (X.T).dot(F)

  dW += reg * W

  dW /= N

  return loss, dW


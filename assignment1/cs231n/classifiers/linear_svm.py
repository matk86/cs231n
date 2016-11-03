import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, y[i]] -= X[i]
        dW[:, j] += X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W*W)
  dW += reg * W
  dW /= num_train

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  delta = 1
  N, D = X.shape
  _, C = W.shape

  XW = X.dot(W)

  Z = XW[np.arange(0, N), y]
  A = XW - Z[:, np.newaxis] + delta

  margin = np.where(A > 0, A, 0)
  # set margin = 0 for correct classes
  margin[np.arange(0, N), y] = 0
  loss = np.sum(margin)/N
  loss += 0.5 * reg * np.sum(W * W)

  margin = np.where(margin > 0, 1, 0)

  F = np.zeros((N, C))
  # j!= y_i
  F[margin > 0] = 1
  # j = y_i
  F[np.arange(0, N), y] = -1 * np.sum(margin, axis=1)
  dW = (X.T).dot(F)

  dW += reg * W

  dW /= N

  return loss, dW

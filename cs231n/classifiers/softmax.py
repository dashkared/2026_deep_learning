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
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss
        
        for k in range(num_classes):
            if k == y[i]:
                dW[:, k] += (p[k] - 1) * X[i]
            else:
                dW[:, k] += p[k] * X[i]
                
    dW /= num_train
        

    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################

    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    scores = X.dot(W) # shape: (N, C)
    scores = scores - np.max(scores, axis=1, keepdims=True) # for numerical stability
    p = np.exp(scores) # shape: (N, C)
    p = p / np.sum(p, axis=1, keepdims=True) # normalize to get probabilities

    correct_probs = p[np.arange(num_train), y] # shape: (N,)
    
    loss = -np.sum(np.log(correct_probs)) / num_train + reg * np.sum(W * W) # shape: (N,)
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################

    dscodes = p.copy() # shape: (N, C)
    dscodes[np.arange(num_train), y] -= 1 # subtract 1 for the correct class
    dW = X.T.dot(dscodes) / num_train + 2 * reg * W # shape: (D, C)

    return loss, dW

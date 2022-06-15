"""
Define optimization objectives
"""

import numpy as np
from scipy.special import expit as sigmoid
from scipy.special import logsumexp, softmax
import scipy
import scipy.sparse
import scipy.sparse.linalg
from typing import Sequence


def norm(M, **kwargs):
    if scipy.sparse.issparse(M):
        return scipy.sparse.linalg.norm(M, **kwargs)
    else:
        return scipy.linalg.norm(M, **kwargs)


def multiclass_log_loss(w, X, y):
    s = (X @ w)
    s_true = s[np.arange(X.shape[0]), y]
    loss = (logsumexp(s, axis=1) - s_true).mean(0)
    p_softmax = softmax(s, axis=1)
    p_softmax[np.arange(X.shape[0]), y] -= 1
    grad = X.T @ p_softmax / X.shape[0]
    return loss, grad


def binary_logistic_loss(w, X, y):
    s = -y * (X @ w)
    loss = np.logaddexp(0, s).mean(0)
    grad = (-y * sigmoid(s)) @ X / X.shape[0]
    return loss, grad


def square_loss(w, X, y):
    err = X @ w - y
    loss = 0.5 * (err ** 2).mean(0)
    grad = err @ X / X.shape[0]
    return loss, grad


class FiniteSumObjective:
    def __init__(self, data, loss='logistic', l2_reg=0.0):
        self.X, self.y = data
        self.loss = loss
        self.l2_reg = l2_reg

        if loss == 'logistic':
            self._assert_binary_labels()
            self.loss_and_gradient = binary_logistic_loss
            self._smoothness = norm(self.X, ord=2, axis=1).max() / 4
            self._argument_shape = (self.X.shape[1], )
        elif loss == 'multiclass_log':  # warning! this is not tested
            # turns labels into categorical labels
            _, self.y = np.unique(self.y, return_inverse=True)
            self.loss_and_gradient = multiclass_log_loss
            self.num_labels = self.y.max() + 1
            self._smoothness = norm(self.X, ord=2, axis=1).max() / 4  # need to double-check this
            self._argument_shape = (self.X.shape[1], self.num_labels)
        elif loss == 'square_binary':
            self._assert_binary_labels()  # this is to ensure labels in {-1,1}
            self.loss_and_gradient = square_loss
            self._smoothness = norm(self.X, ord=2, axis=1).max()
            self._argument_shape = (self.X.shape[1], )
        else:
            raise ValueError(f'Unknown loss type {loss}')

        self._X_dense = None

    def eval_full(self, w):
        return self.eval_inds(w, slice(None))

    def eval_inds(self, w, i):
        if not isinstance(i, (Sequence, np.ndarray, slice)):
            i = [i]  # to make indexing consistent with a full batch

        loss, grad = self.loss_and_gradient(w, self.X[i], self.y[i])
        if self.l2_reg > 0:
            loss += self.l2_reg / 2 * (w ** 2).sum()
            grad += self.l2_reg * w

        return loss, grad

    def _assert_binary_labels(self):
        labels = tuple(np.unique(self.y))
        if labels == (-1, 1):
            pass
        elif labels == (0, 1):
            self.y = 2 * self.y - 1
        elif labels == (1, 2):
            self.y = 2 * self.y - 3
        else:
            raise ValueError(f'Label values are {labels}; this is not consistent with binary classification')

    @property
    def smoothness(self):
        return self._smoothness

    @property
    def argument_shape(self):
        return self._argument_shape

    @property
    def n(self):
        return self.X.shape[0]

    @property
    def X_dense(self):
        if self._X_dense is None:
            if scipy.sparse.issparse(self.X):
                self._X_dense = self.X.toarray()
            else:
                self._X_dense = self.X

        return self._X_dense




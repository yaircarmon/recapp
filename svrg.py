"""
Implementation of SVRG following
"Accelerating Stochastic Gradient Descent using Predictive Variance Reduction"
by Rie Johnson and Tong Zhang, NeurIPS 2013.
"""

import numpy as np
from numpy.linalg import norm
from objectives import FiniteSumObjective

import pandas as pd
import datetime

from loguru import logger
from dataclasses import dataclass

import numba


@dataclass
class SvrgConfig:
    step_size: float = 1.0  # in units of 1 / L
    epoch_length: float = 2.0  # how many data passes to do before recentering svrg estimator
    tail_averaging_fraction: float = 0.5  # fraction of iterations to average when computing output


def svrg(obj: FiniteSumObjective, complexity_budget, config: SvrgConfig):
    x = np.zeros(obj.argument_shape)

    n_epochs = int(complexity_budget / (1 + config.epoch_length))
    budget_consumed = 0

    results = []

    logger.info(f'Starting SVRG with complexity budget {complexity_budget} and config {config}')
    for epoch in range(1, n_epochs + 1):
        x = one_epoch_svrg(obj, x,
                           step_size=config.step_size,
                           epoch_length=config.epoch_length,
                           tail_averaging_fraction=config.tail_averaging_fraction)
        f_at_x, grad_at_x = obj.eval_full(x)  # this is for logging purposes, so we are not counting this against the
        # complexity budget (we can also extract this value from one_epoch_svrg by being more careful,
        # but this is not likely to be a compute bottleneck)
        budget_consumed += 1 + config.epoch_length
        results.append(dict(epoch=epoch, budget_consumed=budget_consumed, timestamp=datetime.datetime.now(),
                            f_value=f_at_x, grad_norm=norm(grad_at_x), x_norm=norm(x)))
        logger.info(f'Ran epoch {epoch}/{n_epochs}. Objective = {f_at_x:.3g}, grad norm = {norm(grad_at_x):.3g}, '
                    f'budget consumed = {budget_consumed}')

    return x, pd.DataFrame(results)


def one_epoch_svrg(obj: FiniteSumObjective, sgd_init, vr_center=None, prox_center=None,
                   prox_lambda=0.0, step_size=1.0, epoch_length=2.0,
                   tail_averaging_fraction=0.5):
    m = int(obj.n * epoch_length)
    tail_averaging_m = max(1, int(tail_averaging_fraction * m))  # tail_averaging_fraction = 0 corresponds to to returning the last iterate
    eta = step_size / obj.smoothness

    if vr_center is None:
        vr_center = sgd_init
    f_at_vr_center, grad_at_center = obj.eval_full(vr_center)

    if prox_center is None:
        prox_center = sgd_init

    x = sgd_init.copy()
    x_av = np.zeros_like(x)

    sample_inds = np.random.choice(obj.n, m)

    if obj.loss == 'logistic':
        # very fast numba-compiled implementation
        if obj.X.shape[1] < 1000:  # heuristic for whether to use dense or sparse representation
            bA = (obj.y.reshape(-1, 1) * obj.X_dense)[sample_inds]
            x_av = efficient_logistic_regression_svrg(bA, x, vr_center, prox_center, grad_at_center,
                                                      eta, prox_lambda, obj.l2_reg, m - tail_averaging_m)
        else:
            x_av = efficient_sparse_logistic_regression_svrg(sample_inds, obj.y, obj.X.data, obj.X.indices, obj.X.indptr,
                                                      x, vr_center, prox_center, grad_at_center,
                                                      eta, prox_lambda, obj.l2_reg, m - tail_averaging_m)
    else:
        # slow generic implementation
        for i, ind in enumerate(sample_inds):
            _, g = obj.eval_inds(x, ind)
            _, gc = obj.eval_inds(vr_center, ind)
            vr_g = grad_at_center + (g - gc)
            x = (x + eta * prox_lambda * prox_center - eta * vr_g) / (1 + eta * prox_lambda)
            if i >= m - tail_averaging_m:
                x_av += x
        x_av /= tail_averaging_m

    return x_av


@numba.njit()
def efficient_logistic_regression_svrg(bA, x, x_vr, x_prox, grad_x_vr,
                                       eta, prox_lambda, l2_reg, tail_averaging_thresh):
    x_sum = 0 * x
    for i in range(bA.shape[0]):
        bAi = bA[i]
        g = -bAi * (1 / (1 + np.exp(bAi @ x))) + l2_reg * x
        gc = -bAi * (1 / (1 + np.exp(bAi @ x_vr))) + l2_reg * x_vr

        ghat = grad_x_vr + g - gc
        x = (x + eta * prox_lambda * x_prox - eta * ghat) / (1 + eta * prox_lambda)
        if i >= tail_averaging_thresh:
            x_sum += x

    return x_sum / (bA.shape[0] - tail_averaging_thresh)


@numba.njit()
def efficient_sparse_logistic_regression_svrg(
        sample_inds, b, A_data, A_indices, A_indptr,
        x, x_vr, x_prox, grad_x_vr,
        eta, prox_lambda, l2_reg, tail_averaging_thresh):
    x_sum = 0 * x
    for i, j in enumerate(sample_inds):
        inds = A_indices[A_indptr[j]:A_indptr[j + 1]]

        bAi = b[j] * A_data[A_indptr[j]:A_indptr[j + 1]]
        g = l2_reg * x
        g[inds] -= bAi * (1 / (1 + np.exp(bAi @ x[inds])))
        gc = l2_reg * x_vr
        gc[inds] -= bAi * (1 / (1 + np.exp(bAi @ x_vr[inds])))

        ghat = grad_x_vr + g - gc
        x = (x + eta * prox_lambda * x_prox - eta * ghat) / (1 + eta * prox_lambda)
        if i >= tail_averaging_thresh:
            x_sum += x

    return x_sum / (len(sample_inds) - tail_averaging_thresh)

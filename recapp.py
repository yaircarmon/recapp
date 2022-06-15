"""
Implementation of RECAPP following
"RECAPP: Crafting a More Efficient Catalyst for Convex Optimization"
by Yair Carmon, Arun Jambulapati, Yujia Jin and Aaron Sidford, ICML 2022.
"""
import numpy as np
from numpy.linalg import norm
from objectives import FiniteSumObjective

import pandas as pd
import datetime
import scipy

from loguru import logger
from dataclasses import dataclass

from svrg import one_epoch_svrg


@dataclass
class RecappConfig:
    prox_lambda: float = 1.0  # in units of L / n
    mlmc_p: float = 0.0  # the parameter p in Alg. 2; a value of 0 indicates we don't use MLMC
    mlmc_minimum_svrg_calls: int = 1  # corresponds to j_0 + 1 in the paper
    svrg_step_size: float = 1.0  # in units of 1 / L
    svrg_epoch_length: float = 2.0  # how many data passes to do before recentering svrg estimator
    svrg_tail_averaging_fraction: float = 0.5  # fraction of iterations to average when computing output
    warmstart: bool = True  # whether to use the warmstart procedure


def recapp(obj: FiniteSumObjective, complexity_budget, config: RecappConfig):
    prox_lambda = config.prox_lambda * obj.smoothness / obj.n
    logger.info(f'Number of functions in the finite sum objective = {obj.n}')

    alpha = 1.0

    budget_consumed = 0
    recapp_iter = 0

    results = []

    logger.info(f'Starting RECAPP with complexity budget {complexity_budget} and config {config}')

    if config.warmstart:
        x, ws_budget = warmstart_svrg(obj, np.zeros(obj.argument_shape), tail_averaging_fraction=config.svrg_tail_averaging_fraction)
        v = x
        budget_consumed += ws_budget
    else:
        x = np.zeros(obj.argument_shape)
        v = np.zeros(obj.argument_shape)

    while budget_consumed < complexity_budget:
        alpha_next = (- alpha ** 2 + np.sqrt(alpha ** 4 + 4 * alpha ** 2)) / 2
        s = (1 - alpha_next) * x + alpha_next * v
        x_next, xtilde_next, budget_per_mlmc = MLMC_svrg(obj,
                                                         vr_center=x,
                                                         prox_center=s,
                                                         prox_lambda=prox_lambda,
                                                         step_size=config.svrg_step_size,
                                                         epoch_length=config.svrg_epoch_length,
                                                         mlmc_p=config.mlmc_p,
                                                         minimum_svrg_calls=config.mlmc_minimum_svrg_calls,
                                                         tail_averaging_fraction=config.svrg_tail_averaging_fraction)

        budget_consumed += budget_per_mlmc
        v_next = v - (s - xtilde_next) / alpha_next

        f_at_x, grad_at_x = obj.eval_full(x)  # this is for logging purposes
        x, v, alpha = x_next, v_next, alpha_next

        recapp_iter += 1

        results.append(dict(epoch=recapp_iter, budget_consumed=budget_consumed, timestamp=datetime.datetime.now(),
                            f_value=f_at_x, grad_norm=norm(grad_at_x), x_norm=norm(x)))
        logger.info(f'Ran epoch {recapp_iter}. Objective = {f_at_x:.3g}, grad norm = {norm(grad_at_x):.3g}, '
                    f'budget consumed = {budget_consumed}. A_t is {alpha_next ** (-2)}.')
    return x, pd.DataFrame(results)


def MLMC_svrg(obj: FiniteSumObjective, vr_center=None, prox_center=None, mlmc_p=0.0, minimum_svrg_calls=1,
              prox_lambda=0.0, step_size=1.0, epoch_length=2.0,
              tail_averaging_fraction=0.5):
    mlmc_1_minus_p = 1 - mlmc_p
    rv_geom = scipy.stats.geom.rvs(mlmc_1_minus_p) - 1  # takes values in 0, 1, 2, ...
    prob_rv_geom = (mlmc_p ** rv_geom) * mlmc_1_minus_p
    tot_svrg_calls = minimum_svrg_calls + rv_geom
    x = x_previous = x_init = prox_center
    budget_consumed = 0

    # the following setting ensures that on average the total oracle complexity is 1 + epoch_length
    inner_epoch_length = ((1 + epoch_length) / (minimum_svrg_calls + mlmc_p / (1 - mlmc_p))) - 1
    if inner_epoch_length < 0:
        raise ValueError('epoch_length parameter supplied to MLMC_svrg is too small for given value of '
                         'minimum_svrg_calls and mlmc_p')

    for i in range(tot_svrg_calls):
        if i == minimum_svrg_calls - 1:
            x_init = x.copy()
        if i == tot_svrg_calls - 1:
            x_previous = x.copy()
        x = one_epoch_svrg(obj, x, vr_center=vr_center if i == 0 else x,
                           prox_center=prox_center,
                           prox_lambda=prox_lambda,
                           step_size=step_size,
                           epoch_length=inner_epoch_length,
                           tail_averaging_fraction=tail_averaging_fraction)
        budget_consumed += 1 + inner_epoch_length

    x_tilde = x_init + (x - x_previous) / prob_rv_geom
    return x, x_tilde, budget_consumed


def warmstart_svrg(obj: FiniteSumObjective, sgd_init, tail_averaging_fraction=0.5):
    x = sgd_init
    budget_consumed = 0
    num_epochs = int(np.floor(np.log2(np.log2(obj.n))))
    for i in range(num_epochs):
        eta = obj.n ** (-2 ** (-i-1))
        epoch_length = 1.0
        x = one_epoch_svrg(obj, x, step_size=eta, epoch_length=epoch_length, tail_averaging_fraction=tail_averaging_fraction)
        budget_consumed += 1 + epoch_length
        f_at_x, grad_at_x = obj.eval_full(x)
        logger.info(
            f'Warm start epoch {i + 1}/{num_epochs}. Objective = {f_at_x:.3g}, grad norm = {norm(grad_at_x):.3g}, '
            f'budget consumed = {budget_consumed}')
    return x, budget_consumed

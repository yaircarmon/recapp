"""
Implementation of Catalyst following
"Catalyst Acceleration for First-order Convex Optimization: from Theory to Practice"
by Hongzhou Lin, Julien Mairal and Zaid Harchaoui, JMLR 2018
"""
import numpy as np
from numpy.linalg import norm
from objectives import FiniteSumObjective

import pandas as pd
import datetime

from loguru import logger
from dataclasses import dataclass

from svrg import one_epoch_svrg

from recapp import warmstart_svrg


@dataclass
class CatalystConfig:
    prox_lambda: float = 1.0  # in units of L / n
    stopping_criterion: str = 'c1'  # supported: c1 from Catalyst paper
    prox_init: str = 'c3'  # to be supported: 'x', 'y', 'c1', 'c3' (Catalyst C1* corresponds to stopping_criterion='c1' and prox_init='c3')
    strong_convexity: float = 0.0  # a negative value indicates taking the l2_reg value from the objective
    svrg_step_size: float = 1.0  # in units of 1 / L
    svrg_epoch_length: float = 2.0  # how many data passes to do before recentering svrg estimator
    svrg_tail_averaging_fraction: float = 0.5  # whether or not to use the warmstart procedure
    warmstart: bool = False  # whether or not to use a warm start a la RECAPP


def catalyst(obj: FiniteSumObjective, complexity_budget, config: CatalystConfig):
    if config.strong_convexity < 0:
        strong_convexity = obj.l2_reg
    else:
        strong_convexity = config.strong_convexity

    prox_lambda = config.prox_lambda * obj.smoothness / obj.n

    budget_consumed = 0
    if config.warmstart:
        x, ws_budget = warmstart_svrg(obj, np.zeros(obj.argument_shape))
        budget_consumed += ws_budget
    else:
        x = np.zeros(obj.argument_shape)

    y = x
    y_old = x
    f_init, grad_init = obj.eval_full(x)
    q = strong_convexity / (strong_convexity + prox_lambda)
    if strong_convexity > 0:
        alpha = np.sqrt(q)
    else:
        alpha = 1
    catalyst_iter = 0

    results = []

    logger.info(f'Starting Catalyst with complexity budget {complexity_budget} and config {config}')

    while budget_consumed < complexity_budget:
        # define starting point of each prox oracle based on prox_init
        if config.prox_init == 'x':
            prox_init = x
        elif config.prox_init == 'y':
            prox_init = y
        elif config.prox_init == 'c1':
            prox_init = x + prox_lambda / (prox_lambda + strong_convexity) * (y - y_old)
        elif config.prox_init == 'c3':
            prox_init_1 = x
            prox_init_2 = x + prox_lambda / (prox_lambda + strong_convexity) * (y - y_old)
            f_at_x_1, grad_at_x_1 = obj.eval_full(prox_init_1)
            f_at_x_2, grad_at_x_2 = obj.eval_full(prox_init_2)
            budget_consumed += 1
            if f_at_x_1 + prox_lambda * norm(prox_init_1 - y) ** 2 / 2 > f_at_x_2 + prox_lambda * norm(
                    prox_init_2 - y) ** 2 / 2:
                prox_init = prox_init_2
            else:
                prox_init = prox_init_1
        else:
            raise ValueError('Undefined way of initializing prox oracle!')

        # implementation of prox oracle using svrg
        if config.stopping_criterion == 'c1':
            x_next, cost_of_svrg = multi_epoch_svrg(obj, prox_init, f_init, catalyst_iter,
                                                    strong_convexity=strong_convexity,
                                                    prox_center=y,
                                                    prox_lambda=prox_lambda,
                                                    step_size=config.svrg_step_size,
                                                    epoch_length=config.svrg_epoch_length,
                                                    tail_averaging_fraction=config.svrg_tail_averaging_fraction)
            budget_consumed += cost_of_svrg
        else:
            raise ValueError('Undefined stopping criterion!')

        alpha_next = (q - alpha ** 2 + np.sqrt((q - alpha ** 2) ** 2 + 4 * alpha ** 2)) / 2
        beta = (alpha * (1 - alpha)) / (alpha ** 2 + alpha_next)
        y_old = y
        y = x_next + beta * (x_next - x)
        x, alpha = x_next, alpha_next
        catalyst_iter += 1
        f_at_x, grad_at_x = obj.eval_full(x)  # this is for logging purposes, so we are not counting this against the
        # complexity budget (we can also extract this value from one_epoch_svrg by being more careful,
        # but this is not likely to be a compute bottleneck)
        results.append(dict(epoch=catalyst_iter, budget_consumed=budget_consumed, timestamp=datetime.datetime.now(),
                            f_value=f_at_x, grad_norm=norm(grad_at_x), x_norm=norm(x)))
        logger.info(f'Ran epoch {catalyst_iter}. Objective = {f_at_x:.3g}, grad norm = {norm(grad_at_x):.3g}, '
                    f'budget consumed = {budget_consumed}.')

    return x, pd.DataFrame(results)


def multi_epoch_svrg(obj: FiniteSumObjective, svrg_init, value_init, iter, strong_convexity=0, prox_center=None,
                     prox_lambda=0.0, step_size=1.0, epoch_length=2.0,
                     tail_averaging_fraction=0.5):
    x = svrg_init
    budget_consumed = 0

    while True:
        x = one_epoch_svrg(obj, x,
                           prox_center=prox_center,
                           prox_lambda=prox_lambda,
                           step_size=step_size,
                           epoch_length=epoch_length,
                           tail_averaging_fraction=tail_averaging_fraction)
        budget_consumed += 1 + epoch_length
        # do the check after svrg step to ensure we at least do one of those
        f_at_x, grad_at_x = obj.eval_full(x)
        opt_gap_bound = norm(grad_at_x + prox_lambda * (x - prox_center)) ** 2 / (2 * (strong_convexity + prox_lambda))
        if opt_gap_bound <= catalyst_stopping_c1(value_init, iter, strong_convexity, prox_lambda):
            break

    budget_consumed += 1  # for the final eval when the loop breaks

    return x, budget_consumed


def catalyst_stopping_c1(value_init, iter, strong_convexity=0, prox_lambda=0.0):
    if strong_convexity == 0:
        error = value_init / (2 * (iter + 1) ** (4.1))
    else:
        rho = 0.9 * np.sqrt(strong_convexity / (strong_convexity + prox_lambda))
        error = 0.5 * (1 - rho) ** iter * value_init
    return error


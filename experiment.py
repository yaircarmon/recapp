"""
Run a single experiment finite-sum optimization experiment
"""
import os
from loguru import logger
import yaml
from pathlib import Path

import numpy as np
import pandas as pd

from simple_parsing import ArgumentParser

from dataclasses import dataclass, asdict

from objectives import FiniteSumObjective
from datasets import load_data
from svrg import svrg, SvrgConfig
from catalyst import catalyst, CatalystConfig
from recapp import recapp, RecappConfig

ALG_TO_FUN = dict(svrg=svrg, catalyst=catalyst, recapp=recapp)
ALG_TO_CFG = dict(svrg=SvrgConfig, catalyst=CatalystConfig, recapp=RecappConfig)
CFG_TO_ALG = {v: k for k, v in ALG_TO_CFG.items()}


@dataclass
class Config:
    algorithm: str = 'svrg'  # do NOT set this via command line; write the algorithm name as the first argument instead
    algorithm_config: object = SvrgConfig()  # do NOT set this via command line; algorithm-speicific configurations are automagically added as command line arguments

    # bookkeeping
    output_dir: Path = ''  # directory to write outputs to
    log_to_stderr: bool = False
    data_root: Path = Path('data')  # root directory for storing datasets

    # seeds
    seed: int = 42  # random seed

    # data
    dataset: str = 'covtype.binary'  # see the libsvmdata python package for list of supported dataset names
    data_n: int = -1  # if positive, truncate dataset to first n entries

    # objective
    loss_type: str = 'logistic'  # supported: 'logistic', 'multiclass_log', and 'square_binary'. Note: only the first loss (logistic regression) is currently thoroughly tested and accelerated with Numba
    loss_l2_reg: float = 0.0  # amount of l2 regulariztion

    complexity_budget: float = 100  # in units of epochs


def run_experiment(config: Config):
    # test that algorithm and algorithm-specific configurations make sense
    if config.algorithm not in ALG_TO_CFG:
        raise ValueError(f'Specified algorithm "{config.algorithm}" is not in the list of supported '
                         f'algorithms: {ALG_TO_CFG.keys()}')
    if not isinstance(config.algorithm_config, ALG_TO_CFG[config.algorithm]):
        raise ValueError(f'Algorithm configuration object type ({type(config.algorithm_config)}) does not match '
                         f'selected algorithm ({config.algorithm})')

    # set up output dir and logging
    output_dir = config.output_dir
    if isinstance(output_dir, str) and output_dir != '':
        output_dir = Path(output_dir)
        write_output = True
    else:
        write_output = False
    if not config.log_to_stderr:
        logger.remove()
    if write_output:
        os.makedirs(output_dir, exist_ok=True)
        logger.add(output_dir / 'experiment.log')
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump({k: v if not isinstance(v, Path) else str(v) for k, v in asdict(config).items()}, f)

    logger.info(f'Starting experiment with config {config}')

    # set seed
    if config.seed is not None:
        np.random.seed(config.seed)  #todo: set seed in a more modern way

    # set up data
    logger.info(f'Loading dataset {config.dataset}')
    data = load_data(config.dataset, n=config.data_n, root=config.data_root)

    # set up objective
    logger.info(f'Setting up {config.loss_type} loss')
    objective = FiniteSumObjective(data, loss=config.loss_type, l2_reg=config.loss_l2_reg)

    # run the algorithm
    logger.info(f'Starting algorithm {config.algorithm}')
    algorithm_function = ALG_TO_FUN[config.algorithm]
    sol, summary = algorithm_function(objective, config.complexity_budget, config.algorithm_config)

    # save dataframe containing summary of run to a csv file
    #todo: save results intermittently during the run
    if write_output:
        summary.to_csv(output_dir / 'summary.csv')

    return sol, summary


if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()
    for algorithm, cfg_class in ALG_TO_CFG.items():
        subparser = subparsers.add_parser(algorithm)
        subparser.add_arguments(Config, dest='config')
        subparser.add_arguments(cfg_class, dest='algorithm_config')
    args = parser.parse_args()

    config = args.config
    config.algorithm = CFG_TO_ALG[type(args.algorithm_config)]
    config.algorithm_config = args.algorithm_config

    sol, summary = run_experiment(config)
    # print(f'Elapsed time = {summary.timestamp.iloc[-1] - summary.timestamp.iloc[0]}')

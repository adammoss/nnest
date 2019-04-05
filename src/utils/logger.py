import logging
import sys
import os
import numpy as np
import pandas as pd
import errno


def create_logger(module_name, level=logging.INFO):
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter('[{}] [%(levelname)s] %(message)s'.format(module_name))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def save_ess(ess, path):
    df = pd.DataFrame(np.reshape(ess, [1, -1]))
    df.to_csv('{}/ess.csv'.format(path), mode='a', header=False)


def ensure_directory(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def make_run_dir(log_dir, run_num=None, append_run_num=True):
    """Generates a new numbered directory for this run to store output"""
    try:
        os.makedirs(log_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    if run_num is None or run_num == '':
        run_num = (sum(os.path.isdir(os.path.join(log_dir,i))
                      for i in os.listdir(log_dir)) + 1)
    if append_run_num:
        run_dir = os.path.join(log_dir, 'run%s' % run_num)
    else:
        run_dir = log_dir
    if not os.path.isdir(run_dir):
        print('Creating directory for new run %s' % run_dir)
        os.makedirs(run_dir)
    if not os.path.isdir(os.path.join(run_dir, 'info')):
        os.makedirs(os.path.join(run_dir, 'info'))
        os.makedirs(os.path.join(run_dir, 'results'))
        os.makedirs(os.path.join(run_dir, 'chains'))
        os.makedirs(os.path.join(run_dir, 'extra'))
        os.makedirs(os.path.join(run_dir, 'plots'))

    return {'run_dir': run_dir,
            'info': os.path.join(run_dir, 'info'),
            'results': os.path.join(run_dir, 'results'),
            'chains': os.path.join(run_dir, 'chains'),
            'extra': os.path.join(run_dir, 'extra'),
            'plots': os.path.join(run_dir, 'plots')
            }

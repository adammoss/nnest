"""
.. module:: mcmc
   :synopsis: Sampler base class
.. moduleauthor:: Adam Moss <adam.moss@nottingham.ac.uk>
"""

from __future__ import print_function
from __future__ import division

import os
import json
import logging
import numpy as np

from nnest.trainer import Trainer
from nnest.utils.logger import create_logger, make_run_dir


class Sampler(object):

    def __init__(self,
                 x_dim,
                 loglike,
                 transform=None,
                 append_run_num=True,
                 run_num=None,
                 hidden_dim=128,
                 num_slow=0,
                 num_derived=0,
                 batch_size=100,
                 flow='nvp',
                 num_blocks=5,
                 num_layers=2,
                 log_dir='logs/test',
                 resume=True,
                 use_gpu=False,
                 base_dist=None,
                 scale='',
                 trainer=None
                 ):

        self.x_dim = x_dim
        self.num_derived = num_derived
        self.num_params = x_dim + num_derived

        assert x_dim > num_slow
        self.num_slow = num_slow
        self.num_fast = x_dim - num_slow
        
        def safe_loglike(x):
            if isinstance(x, list):
                x = np.array(x)
            if len(x.shape) == 1:
                assert x.shape[0] == self.x_dim
                x = np.expand_dims(x, 0)
            res = loglike(x)
            if isinstance(res, tuple):
                logl, derived = res
            else:
                logl = res
                # Set derived shape to be (batch size, 0)
                derived = np.array([[] for _ in x])
            if len(logl.shape) == 0:
                logl = np.expand_dims(logl, 0)
            logl[np.logical_not(np.isfinite(logl))] = -1e100
            if len(derived.shape) == 1 or derived.shape[1] != self.num_derived:
                raise ValueError('Is the number of derived parameters correct and derived has the correct shape?')
            return logl, derived

        self.loglike = safe_loglike

        if transform is None:
            self.transform = lambda x: x
        else:
            def safe_transform(x):
                if isinstance(x, list):
                    x = np.array(x)
                if len(x.shape) == 1:
                    assert x.shape[0] == self.x_dim
                    x = np.expand_dims(x, 0)
                return transform(x)
            self.transform = safe_transform

        self.use_mpi = False
        try:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.mpi_size = self.comm.Get_size()
            self.mpi_rank = self.comm.Get_rank()
            if self.mpi_size > 1:
                self.use_mpi = True
        except:
            self.mpi_size = 1
            self.mpi_rank = 0

        self.log = not self.use_mpi or (self.use_mpi and self.mpi_rank == 0)

        args = locals()
        args.update(vars(self))

        if self.log:
            self.logs = make_run_dir(log_dir, run_num, append_run_num=append_run_num)
            log_dir = self.logs['run_dir']
            self._save_params(args)
        else:
            log_dir = None

        self.resume = resume

        self.logger = create_logger(__name__, level=logging.INFO)

        if trainer is None:
            self.trainer = Trainer(
                x_dim,
                hidden_dim,
                num_slow=num_slow,
                batch_size=batch_size,
                flow=flow,
                num_blocks=num_blocks,
                num_layers=num_layers,
                log_dir=log_dir,
                log=self.log,
                use_gpu=use_gpu,
                base_dist=base_dist,
                scale=scale)
        else:
            self.trainer = trainer

        if self.log:
            self.logger.info('Num base params [%d]' % (self.x_dim))
            self.logger.info('Num derived params [%d]' % (self.num_derived))
            self.logger.info('Total params [%d]' % (self.num_params))

    def _save_params(self, my_dict):
        my_dict = {k: str(v) for k, v in my_dict.items()}
        with open(os.path.join(self.logs['info'], 'params.txt'), 'w') as f:
            json.dump(my_dict, f, indent=4)

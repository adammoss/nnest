"""
.. module:: mcmc
   :synopsis: MCMC sampler
.. moduleauthor:: Adam Moss <adam.moss@nottingham.ac.uk>
Performs MCMC to obtain posterior samples
"""

from __future__ import print_function
from __future__ import division

import logging

import numpy as np
from getdist.mcsamples import MCSamples

from nnest.sampler import Sampler


class MCMCSampler(Sampler):

    def __init__(self,
                 x_dim,
                 loglike,
                 prior=None,
                 append_run_num=True,
                 hidden_dim=16,
                 num_slow=0,
                 num_derived=0,
                 batch_size=100,
                 flow='spline',
                 num_blocks=3,
                 num_layers=1,
                 learning_rate=0.001,
                 log_dir='logs/test',
                 base_dist=None,
                 scale='',
                 use_gpu=False,
                 trainer=None,
                 transform_prior=True,
                 oversample_rate=-1,
                 log_level=logging.INFO):
        """

        Args:
            x_dim:
            loglike:
            prior:
            append_run_num:
            hidden_dim:
            num_slow:
            num_derived:
            batch_size:
            flow:
            num_blocks:
            num_layers:
            learning_rate:
            log_dir:
            base_dist:
            scale:
            use_gpu:
            trainer:
            transform_prior:
            oversample_rate:
            log_level:
        """

        super(MCMCSampler, self).__init__(x_dim, loglike, append_run_num=append_run_num,
                                          hidden_dim=hidden_dim, num_slow=num_slow,
                                          num_derived=num_derived, batch_size=batch_size, flow=flow,
                                          num_blocks=num_blocks, num_layers=num_layers, learning_rate=learning_rate,
                                          log_dir=log_dir, use_gpu=use_gpu, base_dist=base_dist, scale=scale,
                                          trainer=trainer, prior=prior, transform_prior=transform_prior,
                                          log_level=log_level)

        self.sampler = 'mcmc'
        self.oversample_rate = oversample_rate if oversample_rate > 0 else self.num_fast / self.x_dim

    def run(self):

        pass
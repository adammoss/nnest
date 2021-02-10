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

from nnest.ensemble import EnsembleSampler


class MCMCSampler(EnsembleSampler):

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
                                          log_level=log_level, oversample_rate=oversample_rate)

        self.sampler = 'mcmc'

    def run(
            self,
            mcmc_steps,
            num_chains,
            bootstrap_num_walkers=100,
            bootstrap_mcmc_steps=20,
            bootstrap_burn_in=20,
            bootstrap_iters=1,
            bootstrap_thin=10,
            stats_interval=100,
            output_interval=None,
            initial_jitter=0.01,
            final_jitter=0.01):
        """

        Args:
            mcmc_steps:
            num_chains:
            bootstrap_num_walkers:
            bootstrap_mcmc_steps:
            bootstrap_burn_in:
            bootstrap_iters:
            bootstrap_thin:
            stats_interval:
            output_interval:
            initial_jitter:
            final_jitter:

        Returns:

        """

        self.bootstrap(bootstrap_num_walkers, bootstrap_mcmc_steps=bootstrap_mcmc_steps,
                       bootstrap_burn_in=bootstrap_burn_in, bootstrap_iters=bootstrap_iters,
                       bootstrap_thin=bootstrap_thin, stats_interval=stats_interval,
                       output_interval=output_interval, initial_jitter=initial_jitter,
                       final_jitter=final_jitter)

        samples, latent_samples, derived_samples, loglikes, scale, ncall = self._mcmc_sample(
            mcmc_steps, num_chains=num_chains, stats_interval=stats_interval, output_interval=output_interval)

        samples = self.transform(samples)
        if mcmc_steps > 1:
            self._chain_stats(samples)

        self.samples = np.concatenate((samples, derived_samples), axis=2)
        self.latent_samples = latent_samples
        self.loglikes = loglikes

        self.logger.info("ncall: {:d}\n".format(self.total_calls))

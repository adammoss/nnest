"""
.. module:: ensemble
   :synopsis: Ensemble sampler
.. moduleauthor:: Adam Moss <adam.moss@nottingham.ac.uk>
Performs MCMC to obtain posterior samples
"""

from __future__ import print_function
from __future__ import division

import logging

import numpy as np
from getdist.mcsamples import MCSamples

from nnest.sampler import Sampler


class EnsembleSampler(Sampler):

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
            log_level:
        """

        super(EnsembleSampler, self).__init__(x_dim, loglike, append_run_num=append_run_num,
                                              hidden_dim=hidden_dim, num_slow=num_slow,
                                              num_derived=num_derived, batch_size=batch_size, flow=flow,
                                              num_blocks=num_blocks, num_layers=num_layers, learning_rate=learning_rate,
                                              log_dir=log_dir, use_gpu=use_gpu, base_dist=base_dist, scale=scale,
                                              trainer=trainer, prior=prior, transform_prior=transform_prior,
                                              log_level=log_level)

        self.sampler = 'ensemble'

    def bootstrap(
            self,
            num_walkers,
            bootstrap_mcmc_steps=20,
            bootstrap_burn_in=20,
            bootstrap_iters=1,
            bootstrap_thin=10,
            stats_interval=10,
            output_interval=None,
            initial_jitter=0.01,
            final_jitter=0.01):
        """

        Args:
            num_walkers:
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

        def log_prob(x):
            logl, der = self.loglike(x)
            return logl + self.prior(x), der

        if self.sample_prior is not None:
            x_current = self.sample_prior(num_walkers)
        else:
            raise ValueError('Prior does not have sample method')
        try:
            import emcee
        except:
            raise ImportError
        sampler = emcee.EnsembleSampler(num_walkers, self.x_dim, log_prob, moves=[(emcee.moves.KDEMove(), 1.0)])
        state = sampler.run_mcmc(x_current, bootstrap_burn_in)
        self.logger.info('Initial acceptance [%5.4f]' % (np.mean(sampler.acceptance_fraction)))
        self._chain_stats(np.transpose(sampler.get_chain(), axes=[1, 0, 2]))

        # Use state coordinates to train flow
        mean = np.mean(state.coords, axis=0)
        std = np.std(state.coords, axis=0)
        # Normalise samples
        training_samples = (state.coords - mean) / std
        self.transform = lambda x: x * std + mean
        self.trainer.train(training_samples, jitter=initial_jitter)

        init_samples = None
        init_loglikes = None
        init_derived = None

        for it in range(1, bootstrap_iters + 1):

            if bootstrap_iters > 1:
                jitter = initial_jitter + (it - 1) * (final_jitter - initial_jitter) / (bootstrap_iters - 1)
            else:
                jitter = initial_jitter

            samples, latent_samples, derived_samples, loglikes, ncall = self._ensemble_sample(
                bootstrap_burn_in + bootstrap_mcmc_steps, num_walkers, init_samples=init_samples,
                init_loglikes=init_loglikes, init_derived=init_derived, stats_interval=stats_interval,
                output_interval=None)

            # Remember last position and loglikes
            # init_samples = samples[:, -1, :]
            # init_loglikes = loglikes[:, -1]
            # init_derived = derived_samples[:, -1, :]

            samples = self.transform(samples)
            self._chain_stats(samples)

            mc = MCSamples(samples=[samples[i, bootstrap_burn_in:, :].squeeze() for i in range(samples.shape[0])],
                           loglikes=[-loglikes[i, bootstrap_burn_in:].squeeze() for i in range(loglikes.shape[0])])
            single_samples = mc.makeSingleSamples(single_thin=bootstrap_thin)

            mean = np.mean(single_samples, axis=0)
            std = np.std(single_samples, axis=0)
            # Normalise samples
            training_samples = (single_samples - mean) / std
            self.transform = lambda x: x * std + mean
            self.trainer.train(training_samples, jitter=jitter)

    def run(
            self,
            mcmc_steps,
            num_walkers,
            bootstrap_mcmc_steps=20,
            bootstrap_burn_in=20,
            bootstrap_iters=1,
            bootstrap_thin=10,
            stats_interval=10,
            output_interval=None,
            initial_jitter=0.01,
            final_jitter=0.01):
        """

        Args:
            mcmc_steps:
            num_walkers:
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

        self.bootstrap(num_walkers, bootstrap_mcmc_steps=bootstrap_mcmc_steps,
                       bootstrap_burn_in=bootstrap_burn_in, bootstrap_iters=bootstrap_iters,
                       bootstrap_thin=bootstrap_thin, stats_interval=stats_interval,
                       output_interval=output_interval, initial_jitter=initial_jitter,
                       final_jitter=final_jitter)

        samples, latent_samples, derived_samples, loglikes, ncall = self._ensemble_sample(
            mcmc_steps, num_walkers, stats_interval=stats_interval, output_interval=output_interval)

        samples = self.transform(samples)
        if mcmc_steps > 1:
            self._chain_stats(samples)

        self.samples = np.concatenate((samples, derived_samples), axis=2)
        self.latent_samples = latent_samples
        self.loglikes = loglikes

        self.logger.info("ncall: {:d}\n".format(self.total_calls))

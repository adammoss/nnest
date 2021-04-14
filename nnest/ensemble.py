"""
.. module:: ensemble
   :synopsis: Ensemble sampler
.. moduleauthor:: Adam Moss <adam.moss@nottingham.ac.uk>
Performs MCMC to obtain posterior samples
"""

from __future__ import print_function
from __future__ import division

import logging
import os

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
                 oversample_rate=-1,
                 log_level=logging.INFO,
                 param_names=None):
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
            oversample_rate:
            log_level:
        """

        super(EnsembleSampler, self).__init__(x_dim, loglike, append_run_num=append_run_num,
                                              hidden_dim=hidden_dim, num_slow=num_slow,
                                              num_derived=num_derived, batch_size=batch_size, flow=flow,
                                              num_blocks=num_blocks, num_layers=num_layers, learning_rate=learning_rate,
                                              log_dir=log_dir, use_gpu=use_gpu, base_dist=base_dist, scale=scale,
                                              trainer=trainer, prior=prior, transform_prior=transform_prior,
                                              log_level=log_level, oversample_rate=oversample_rate,
                                              param_names=param_names)

        self.sampler = 'ensemble'

    def bootstrap(
            self,
            mcmc_steps,
            num_walkers,
            iters=1,
            thin=10,
            stats_interval=10,
            output_interval=None,
            initial_jitter=0.01,
            final_jitter=0.01,
            init_samples=None,
            moves=None):
        """

        Args:
            num_walkers:
            mcmc_steps:
            iters:
            thin:
            stats_interval:
            output_interval:
            initial_jitter:
            final_jitter:
            init_samples:
            moves:

        Returns:

        """

        def log_prob(x):
            logl, der = self.loglike(x)
            return logl + self.prior(x), der

        if init_samples is None:
            if self.sample_prior is not None:
                init_samples = self.sample_prior(num_walkers)
            else:
                raise ValueError('Prior does not have sample method')
        try:
            import emcee
        except:
            raise ImportError

        if moves is not None:
            ensemble_moves = []
            for k, v in moves.items():
                if k.lower() == 'stretch':
                    ensemble_moves.append((emcee.moves.StretchMove(), v))
                elif k.lower() == 'kde':
                    ensemble_moves.append((emcee.moves.KDEMove(), v))
                elif k.lower() == 'de':
                    ensemble_moves.append((emcee.moves.DEMove(), v))
                elif k.lower() == 'snooker':
                    ensemble_moves.append((emcee.moves.DESnookerMove(), v))
        else:
            ensemble_moves = [(emcee.moves.StretchMove(), 1.0)]

        self.logger.info('Performing initial emcee run with [%d] walkers' % (num_walkers))
        sampler = emcee.EnsembleSampler(num_walkers, self.x_dim, log_prob, moves=ensemble_moves,
                                        backend=emcee.backends.HDFBackend(os.path.join(self.log_dir, 'emcee.h5')))
        state = sampler.run_mcmc(init_samples, mcmc_steps)
        self.logger.info('Initial acceptance [%5.4f]' % (np.mean(sampler.acceptance_fraction)))
        self._chain_stats(np.transpose(sampler.get_chain(), axes=[1, 0, 2]))

        tau = sampler.get_autocorr_time()
        training_samples = sampler.get_chain(discard=int(2 * np.max(tau)), flat=True, thin=int(0.5 * np.min(tau)))

        for it in range(1, iters + 1):

            if iters > 1:
                jitter = initial_jitter + (it - 1) * (final_jitter - initial_jitter) / (iters - 1)
            else:
                jitter = initial_jitter

            mean = np.mean(training_samples, axis=0)
            std = np.std(training_samples, axis=0)
            # Normalise samples
            training_samples = (training_samples - mean) / std
            self.transform = lambda x: x * std + mean
            self.trainer.train(training_samples, jitter=jitter)

            init_samples = None
            init_loglikes = None
            init_derived = None

            samples, latent_samples, derived_samples, loglikes, ncall = self._ensemble_sample(
                mcmc_steps, num_walkers, init_samples=init_samples,
                init_loglikes=init_loglikes, init_derived=init_derived, stats_interval=stats_interval,
                output_interval=output_interval)

            # Remember last position and loglikes
            # init_samples = samples[:, -1, :]
            # init_loglikes = loglikes[:, -1]
            # init_derived = derived_samples[:, -1, :]

            samples = self.transform(samples)
            self._chain_stats(samples)

            mc = MCSamples(samples=[samples[i, :, :].squeeze() for i in range(samples.shape[0])],
                           loglikes=[-loglikes[i, :].squeeze() for i in range(loglikes.shape[0])])
            training_samples = mc.makeSingleSamples(single_thin=thin)

        return training_samples

    def run(
            self,
            mcmc_steps,
            num_walkers,
            training_samples,
            stats_interval=10,
            output_interval=None,
            initial_jitter=0.01,
            final_jitter=0.01,
            init_samples=None):
        """

        Args:
            mcmc_steps:
            num_walkers:
            training_samples:
            stats_interval:
            output_interval:
            initial_jitter:
            final_jitter:
            training_samples:
            init_samples:

        Returns:

        """

        mean = np.mean(training_samples, axis=0)
        std = np.std(training_samples, axis=0)
        # Normalise samples
        training_samples = (training_samples - mean) / std
        self.transform = lambda x: x * std + mean
        self.trainer.train(training_samples, jitter=initial_jitter)

        samples, latent_samples, derived_samples, loglikes, ncall = self._ensemble_sample(
            mcmc_steps, num_walkers, stats_interval=stats_interval, output_interval=output_interval)

        samples = self.transform(samples)
        if mcmc_steps > 1:
            self._chain_stats(samples)

        self.samples = np.concatenate((samples, derived_samples), axis=2)
        self.latent_samples = latent_samples
        self.loglikes = loglikes

        self.logger.info("ncall: {:d}\n".format(self.total_calls))

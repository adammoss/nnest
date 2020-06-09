"""
.. module:: mcmc
   :synopsis: MCMC sampler
.. moduleauthor:: Adam Moss <adam.moss@nottingham.ac.uk>
Performs MCMC to obtain posterior samples
"""

from __future__ import print_function
from __future__ import division

import os
import glob
import logging

import numpy as np
from getdist.mcsamples import MCSamples
from getdist.chains import chainFiles

from nnest.sampler import Sampler
from nnest.nested import NestedSampler
from nnest.utils.buffer import Buffer


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
                 log_dir='logs/test',
                 base_dist=None,
                 scale='',
                 use_gpu=False,
                 trainer=None,
                 transform_prior=True,
                 log_level=logging.INFO,
                 oversample_rate=-1):
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
            log_dir:
            base_dist:
            scale:
            use_gpu:
            trainer:
            oversample_rate:
            transform_prior:
        """

        super(MCMCSampler, self).__init__(x_dim, loglike, append_run_num=append_run_num,
                                          hidden_dim=hidden_dim, num_slow=num_slow,
                                          num_derived=num_derived, batch_size=batch_size, flow=flow,
                                          num_blocks=num_blocks, num_layers=num_layers, log_dir=log_dir,
                                          use_gpu=use_gpu, base_dist=base_dist, scale=scale, trainer=trainer,
                                          prior=prior, transform_prior=transform_prior, log_level=log_level)

        self.sampler = 'mcmc'
        self.oversample_rate = oversample_rate if oversample_rate > 0 else self.num_fast / self.x_dim

    def init_samples(self, num_walkers=100):
        try:
            import emcee
        except:
            raise ImportError
        def transformed_loglike(x):
            return self.loglike(x)[0] + self.prior(x)
        sampler = emcee.EnsembleSampler(num_walkers, self.loglike.x_dim, transformed_loglike)
        p0 = self.prior.sample(num_walkers)
        state = sampler.run_mcmc(p0, 1)
        return state.coords

    def read_samples(self, fileroot, match='', ignore_rows=0.3, thin=1):
        names = ['p%i' % i for i in range(int(self.num_params))]
        labels = [r'x_%i' % i for i in range(int(self.num_params))]
        if match:
            files = glob.glob(os.path.join(fileroot, match))
        else:
            files = chainFiles(fileroot)
        mc = MCSamples(fileroot, names=names, labels=labels, ignore_rows=ignore_rows)
        mc.readChains(files)
        return mc.makeSingleSamples(single_thin=thin)

    def run(
            self,
            mcmc_steps,
            num_walkers=100,
            init_samples=None,
            buffer_size=1000,
            bootstrap_mcmc_steps=20,
            bootstrap_burn_in=100,
            bootstrap_iters=5,
            bootstrap_thin=10,
            num_chains=5,
            step_size=0,
            stats_interval=1000,
            output_interval=500,
            initial_jitter=0.1,
            final_jitter=0.01):
        """

        Args:
            mcmc_steps:
            init_samples:
            buffer_size:
            bootstrap_mcmc_steps:
            bootstrap_burn_in:
            bootstrap_iters:
            bootstrap_thin:
            num_chains:
            step_size:
            stats_interval:
            output_interval:
            init_samples:
            initial_jitter:
            final_jitter:

        Returns:

        """

        if init_samples is None:
            if self.sample_prior is not None:
                init_samples = self.sample_prior(num_walkers)
            else:
                raise ValueError('Prior does not have sample method')

        mean = np.mean(init_samples, axis=0)
        std = np.std(init_samples, axis=0)
        # Normalise samples
        training_samples = (init_samples - mean) / std
        self.transform = lambda x: x * std + mean
        self.trainer.train(training_samples, jitter=final_jitter)

        state = None

        for it in range(bootstrap_iters):

            samples, latent_samples, derived_samples, loglikes, ncall, state = self._emcee_sample(
                bootstrap_mcmc_steps, num_walkers, init_state=state, stats_interval=1)
            self._chain_stats(samples)
            mc = MCSamples(samples=[samples[i, :, :].squeeze() for i in range(samples.shape[0])],
                           loglikes=[-loglikes[i, :].squeeze() for i in range(loglikes.shape[0])],
                           ignore_rows=0.3)
            single_samples = mc.makeSingleSamples()
            mean = np.mean(single_samples, axis=0)
            std = np.std(single_samples, axis=0)
            # Normalise samples
            training_samples = (single_samples - mean) / std
            self.transform = lambda x: x * std + mean
            self.trainer.train(training_samples, jitter=final_jitter)

            print(samples.shape)
            state = samples[:, -1, :]
            state = None

        samples, latent_samples, derived_samples, loglikes, ncall, state = self._emcee_sample(
            mcmc_steps, num_walkers, init_state=state, stats_interval=1)

        samples = self.transform(samples)

        self.samples = samples
        self.latent_samples = latent_samples
        self.loglikes = loglikes

        self.logger.info("ncall: {:d}\n".format(ncall))

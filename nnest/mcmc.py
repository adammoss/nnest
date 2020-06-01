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
import matplotlib

matplotlib.use('Agg')

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

    def init_samples(self):
        transform = lambda x: 5 * x
        sampler = NestedSampler(self.x_dim, self.loglike, transform=transform, num_live_points=500,
                                trainer=self.trainer)
        sampler.run(strategy=['rejection_prior', 'rejection_flow', 'mcmc'], mcmc_steps=2*self.x_dim)
        self.logz = sampler.logz
        mc = MCSamples(samples=sampler.samples, weights=sampler.weights, loglikes=-sampler.loglikes)
        return mc.makeSingleSamples()

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
            init_samples=None,
            buffer_size=1000,
            bootstrap_mcmc_steps=500,
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

        if step_size == 0.0:
            step_size = 1 / self.x_dim ** 0.5

        if self.log:
            self.logger.info('Alpha [%5.4f]' % (step_size))

        buffer = Buffer(max_size=buffer_size)

        if init_samples is None:
            init_samples = self.init_samples()

        if init_samples.shape[0] > buffer_size:
            buffer.insert(init_samples[np.random.choice(init_samples.shape[0], buffer_size, replace=False)])
        else:
            buffer.insert(init_samples)

        self.logger.info('Using [%d] initial samples' % (len(buffer.data)))

        ncall = 0

        for it in range(1, bootstrap_iters + 1):

            if bootstrap_iters > 1:
                jitter = initial_jitter + (it - 1) * (final_jitter - initial_jitter) / (bootstrap_iters - 1)
            else:
                jitter = initial_jitter

            self.logger.info('Bootstrap step [%d]' % (it))

            training_samples = buffer()
            mean = np.mean(training_samples, axis=0)
            std = np.std(training_samples, axis=0)
            # Normalise samples
            training_samples = (training_samples - mean) / std
            self.transform = lambda x: x * std + mean
            self.trainer.train(training_samples, jitter=jitter)

            samples, latent_samples, derived_samples, loglikes, scale, nc = self._mcmc_sample(
                bootstrap_burn_in + bootstrap_mcmc_steps, num_chains=num_chains,
                step_size=step_size, stats_interval=bootstrap_mcmc_steps, dynamic_step_size=True)
            samples = self.transform(samples)
            samples = samples[:, bootstrap_burn_in:, :]
            samples = samples.reshape((samples.shape[0] * samples.shape[1], samples.shape[2]))
            samples = samples[::bootstrap_thin, :]
            buffer.insert(samples)
            ncall += nc
            self.logger.info('Bootstrapping [%d] samples, ncalls [%d] scale [%5.4f]' % (samples.shape[0], ncall, scale))

        training_samples = buffer()
        mean = np.mean(training_samples, axis=0)
        std = np.std(training_samples, axis=0)
        # Normalise samples
        training_samples = (training_samples - mean) / std
        self.transform = lambda x: x * std + mean
        self.trainer.train(training_samples, jitter=final_jitter)

        samples, latent_samples, derived_samples, loglikes, scale, nc = self._mcmc_sample(
            mcmc_steps, num_chains=num_chains, step_size=step_size,
            out_chain=os.path.join(self.logs['chains'], 'chain'), stats_interval=stats_interval,
            output_interval=output_interval)
        samples = self.transform(samples)
        ncall += nc

        self.samples = samples
        self.latent_samples = latent_samples
        self.loglikes = loglikes

        self.logger.info("ncall: {:d}\n".format(ncall))

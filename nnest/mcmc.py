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
import copy

import numpy as np
from getdist.mcsamples import MCSamples
from getdist.chains import chainFiles
import matplotlib

matplotlib.use('Agg')

from nnest.sampler import Sampler
from nnest.utils.buffer import Buffer


class MCMCSampler(Sampler):

    def __init__(self,
                 x_dim,
                 loglike,
                 prior=None,
                 append_run_num=True,
                 run_num=None,
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
                 oversample_rate=-1,
                 transform_prior=True):
        """

        Args:
            x_dim:
            loglike:
            prior:
            append_run_num:
            run_num:
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
                                          run_num=run_num, hidden_dim=hidden_dim, num_slow=num_slow,
                                          num_derived=num_derived, batch_size=batch_size, flow=flow,
                                          num_blocks=num_blocks, num_layers=num_layers, log_dir=log_dir,
                                          use_gpu=use_gpu, base_dist=base_dist, scale=scale, trainer=trainer,
                                          prior=prior, transform_prior=transform_prior)

        self.sampler = 'mcmc'
        self.oversample_rate = oversample_rate if oversample_rate > 0 else self.num_fast / self.x_dim

    def init_samples(self, num_chains=20, steps=50, propose_scale=0.01, temperature=1.0):
        if self.sample_prior is not None:
            x_current = self.sample_prior(num_chains)
        else:
            raise ValueError('Prior does not have sample method')
        loglike_current, _ = self.loglike(x_current)
        loglike_current += self.prior(x_current)
        samples = [copy.copy(x_current)]
        likes = [copy.copy(loglike_current)]
        for i in range(steps):
            x_propose = x_current + propose_scale * np.random.normal(size=(num_chains, self.x_dim))
            loglike_propose, _ = self.loglike(x_propose)
            loglike_propose += self.prior(x_propose)
            ratio = np.ones(num_chains)
            delta_loglike = (loglike_propose - loglike_current) / temperature
            ratio[np.where(delta_loglike < 0)] = np.exp(delta_loglike[np.where(delta_loglike < 0)])
            r = np.random.uniform(low=0, high=1, size=(num_chains,))
            x_current[np.where(ratio > r)] = x_propose[np.where(ratio > r)]
            loglike_current[np.where(ratio > r)] = loglike_propose[np.where(ratio > r)]
            samples.append(copy.copy(x_current))
            likes.append(copy.copy(loglike_current))
        samples = np.array(samples)
        likes = np.array(likes)
        samples = np.transpose(np.array(samples), axes=[1, 0, 2])
        likes = np.transpose(np.array(likes), axes=[1, 0])
        self._chain_stats(samples)
        return samples, likes

    def read_samples(self, fileroot, match='', ignore_rows=0.3, thin=1):
        names = ['p%i' % i for i in range(int(self.num_params))]
        labels = [r'x_%i' % i for i in range(int(self.num_params))]
        if match:
            files = glob.glob(os.path.join(fileroot, match))
        else:
            files = chainFiles(fileroot)
        mc = MCSamples(fileroot, names=names, labels=labels, ignore_rows=ignore_rows)
        mc.readChains(files)
        samples = mc.makeSingleSamples(single_thin=thin)
        return samples

    def run(
            self,
            init_samples=None,
            buffer_size=5000,
            bootstrap_mcmc_steps=500,
            bootstrap_burn_in=100,
            bootstrap_iters=5,
            bootstrap_thin=1,
            mcmc_steps=5000,
            num_chains=5,
            step_size=0,
            stats_interval=200,
            output_interval=100,
            initial_jitter=0.2,
            final_jitter=0.01):
        """

        Args:
            buffer_size:
            bootstrap_mcmc_steps:
            bootstrap_burn_in:
            bootstrap_iters:
            bootstrap_thin:
            mcmc_steps:
            num_chains:
            step_size:
            ignore_rows:
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

        if init_samples.shape[0] > buffer_size:
            buffer.insert(init_samples[np.random.choice(init_samples.shape[0], buffer_size, replace=False)])
        else:
            buffer.insert(init_samples)

        ncall = 0

        for it in range(1, bootstrap_iters + 1):

            if bootstrap_iters > 1:
                jitter = initial_jitter + (it - 1) * (final_jitter - initial_jitter) / (bootstrap_iters - 1)
            else:
                jitter = initial_jitter

            training_samples = buffer()
            mean = np.mean(training_samples, axis=0)
            std = np.std(training_samples, axis=0)
            # Normalise samples
            training_samples = (training_samples - mean) / std
            self.transform = lambda x: x * std + mean
            self.trainer.train(training_samples, jitter=jitter)

            samples, latent_samples, derived_samples, likes, scale, nc = self._mcmc_sample(
                mcmc_steps=bootstrap_burn_in + bootstrap_mcmc_steps, num_chains=num_chains,
                step_size=step_size, stats_interval=bootstrap_mcmc_steps, dynamic_step_size=True)
            samples = self.transform(samples)
            samples = samples[:, bootstrap_burn_in:, :]
            samples = samples.reshape((samples.shape[0] * samples.shape[1], samples.shape[2]))
            samples = samples[::bootstrap_thin, :]
            buffer.insert(samples)
            ncall += nc
            self.logger.info('Bootstrap step [%d], ncalls [%d], nsamples [%d] scale [%5.4f]' %
                             (it, ncall, samples.shape[0], scale))

        training_samples = buffer()
        mean = np.mean(training_samples, axis=0)
        std = np.std(training_samples, axis=0)
        # Normalise samples
        training_samples = (training_samples - mean) / std
        self.transform = lambda x: x * std + mean
        self.trainer.train(training_samples, jitter=final_jitter)

        samples, latent_samples, derived_samples, likes, scale, nc = self._mcmc_sample(
            mcmc_steps=mcmc_steps, num_chains=num_chains, step_size=step_size,
            out_chain=os.path.join(self.logs['chains'], 'chain'), stats_interval=stats_interval,
            output_interval=output_interval)
        samples = self.transform(samples)
        ncall += nc

        self.samples = samples
        self.latent_samples = latent_samples
        self.loglikes = likes

        print("ncall: {:d}\n".format(ncall))

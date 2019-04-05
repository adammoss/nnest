"""
.. module:: mcmc
   :synopsis: MCMC sampler
.. moduleauthor:: Adam Moss <adam.moss@nottingham.ac.uk>
Performs MCMC to obtain posterior samples
"""

from __future__ import print_function
from __future__ import division

import os

import numpy as np
from getdist.mcsamples import MCSamples
from getdist.chains import chainFiles

from src.sampler import Sampler


class MCMCSampler(Sampler):

    def _init_samples(self, mcmc_steps=5000, mcmc_batch_size=5):
        u = 2 * (np.random.uniform(size=(mcmc_batch_size, self.x_dim)) - 0.5)
        v = self.transform(u)
        logl = self.loglike(v)
        samples = []
        likes = []
        for i in range(mcmc_steps):
            du = np.random.standard_normal(u.shape) * 0.1
            u_prime = u + du
            v_prime = self.transform(u_prime)
            log_ratio_1 = np.zeros(mcmc_batch_size)
            prior = np.logical_or(np.abs(u) > 1, np.abs(u_prime) > 1)
            idx = np.where([np.any(p) for p in prior])
            log_ratio_1[idx] = -np.inf
            rnd_u = np.random.rand(mcmc_batch_size)
            ratio = np.clip(np.exp(log_ratio_1), 0, 1)
            mask = (rnd_u < ratio).astype(int)
            logl_prime = np.full(mcmc_batch_size, logl)
            for idx, im in enumerate(mask):
                if im:
                    lp = self.loglike(self.transform(np.expand_dims(u_prime[idx], 0)))
                    if np.isfinite(lp) and rnd_u[idx] < np.clip(np.exp(lp - logl[idx]), 0, 1):
                        logl_prime[idx] = lp
                    else:
                        mask[idx] = 0
            m = mask[:, None]
            u = u_prime * m + u * (1 - m)
            v = v_prime * m + v * (1 - m)
            logl = logl_prime * mask + logl * (1 - mask)
            samples.append(v)
            likes.append(logl)
        samples = np.transpose(np.array(samples), axes=[1, 0, 2])
        loglikes = -np.transpose(np.array(likes), axes=[1, 0])
        weights = np.ones(loglikes.shape)
        self._chain_stats(samples)
        self._save_samples(samples, weights, loglikes)
        names = ['p%i' % i for i in range(int(self.x_dim))]
        labels = [r'x_%i' % i for i in range(int(self.x_dim))]
        files = chainFiles(os.path.join(self.logs['chains'], 'chain'), first_chain=1, last_chain=mcmc_batch_size)
        mc = MCSamples(self.logs['chains'], names=names, labels=labels, ignore_rows=0.3)
        mc.readChains(files)
        return mc

    def _read_samples(self, fileroot):
        names = ['p%i' % i for i in range(int(self.x_dim))]
        labels = [r'x_%i' % i for i in range(int(self.x_dim))]
        files = chainFiles(fileroot)
        mc = MCSamples(fileroot, names=names, labels=labels, ignore_rows=0.3)
        mc.readChains(files)
        return mc

    def run(
            self,
            train_iters=200,
            mcmc_steps=5000,
            bootstrap_iters=2,
            bootstrap_mcmc_steps=5000,
            bootstrap_fileroot=''):

        for t in range(bootstrap_iters):

            if t == 0:
                if bootstrap_fileroot:
                    mc = self._read_samples(bootstrap_fileroot)
                else:
                    mc = self._init_samples()
            else:
                def transform(x):
                    return x * std + mean
                samples, likes, latent, scale, nc = self.trainer.sample(
                    loglike=self.loglike, transform=transform,
                    mcmc_steps=bootstrap_mcmc_steps, alpha=1.0, dynamic=False, show_progress=True)
                samples = transform(samples)
                self._chain_stats(samples)
                loglikes = -np.array(likes)
                weights = np.ones(loglikes.shape)
                mc = MCSamples(samples=[samples[0]], weights=[weights[0]], loglikes=[loglikes[0]], ignore_rows=0.3)

            print(mc.getMargeStats())
            samples = mc.makeSingleSamples(single_thin=10)
            mean = np.mean(samples, axis=0)
            std = np.std(samples, axis=0)
            samples = (samples - mean) / std
            self.trainer.train(samples, max_iters=train_iters, noise=0.01)

        def transform(x):
            return x * std + mean

        samples, likes, latent, scale, nc = self.trainer.sample(
            loglike=self.loglike, transform=transform,
            mcmc_steps=mcmc_steps, alpha=1.0, dynamic=False, show_progress=True,
            out_chain=os.path.join(self.logs['chains'], 'chain'))
        samples = transform(samples)
        self._chain_stats(samples)
        loglikes = -np.array(likes)
        weights = np.ones(loglikes.shape)
        mc = MCSamples(samples=[samples[0]], weights=[weights[0]], loglikes=[loglikes[0]], ignore_rows=0.3)
        print(mc.getMargeStats())

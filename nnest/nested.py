"""
.. module:: nested
   :synopsis: Nested sampler
.. moduleauthor:: Adam Moss <adam.moss@nottingham.ac.uk>
Performs nested sampling to calculate the Bayesian evidence and posterior samples
Some parts are from the Nestle library by Kyle Barbary (https://github.com/kbarbary/nestle)
"""

from __future__ import print_function
from __future__ import division

import os
import csv

import numpy as np

from nnest.sampler import Sampler


class NestedSampler(Sampler):

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
                 use_gpu=False,
                 num_live_points=1000
                 ):

        self.num_live_points = num_live_points
        self.sampler = 'nested'

        super(NestedSampler, self).__init__(x_dim, loglike, transform=transform, append_run_num=append_run_num,
                                            run_num=run_num, hidden_dim=hidden_dim, num_slow=num_slow, 
                                            num_derived=num_derived, batch_size=batch_size, flow=flow,
                                            num_blocks=num_blocks, num_layers=num_layers, log_dir=log_dir,
                                            use_gpu=use_gpu)

        if self.log:
            self.logger.info('Num live points [%d]' % (self.num_live_points))

        if self.log:
            with open(os.path.join(self.logs['results'], 'results.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'acceptance', 'min_ess',
                                 'max_ess', 'jump_distance', 'scale', 'loglstar', 'logz', 'fraction_remain', 'ncall'])

    def run(
            self,
            mcmc_steps=0,
            mcmc_burn_in=0,
            mcmc_batch_size=10,
            max_iters=1000000,
            update_interval=None,
            log_interval=None,
            dlogz=0.5,
            train_iters=50,
            volume_switch=0,
            alpha=0.0,
            noise=-1.0,
            num_test_samples=0,
            test_mcmc_steps=1000,
            test_mcmc_burn_in=0):

        if update_interval is None:
            update_interval = max(1, round(self.num_live_points))
        else:
            update_interval = round(update_interval)
            if update_interval < 1:
                raise ValueError("update_interval must be >= 1")

        if log_interval is None:
            log_interval = max(1, round(0.2 * self.num_live_points))
        else:
            log_interval = round(log_interval)
            if log_interval < 1:
                raise ValueError("log_interval must be >= 1")

        if mcmc_steps <= 0:
            mcmc_steps = 5 * self.x_dim

        if volume_switch <= 0:
            volume_switch = 1 / mcmc_steps

        if alpha == 0.0:
            alpha = 2 / self.x_dim ** 0.5

        if self.log:
            self.logger.info('MCMC steps [%d] alpha [%5.4f] volume switch [%5.4f]' % (mcmc_steps, alpha, volume_switch))

        if self.use_mpi:
            self.logger.info('Using MPI with rank [%d]' % (self.mpi_rank))
            if self.mpi_rank == 0:
                active_u = 2 * (np.random.uniform(size=(self.num_live_points, self.x_dim)) - 0.5)
            else:
                active_u = np.empty((self.num_live_points, self.x_dim), dtype=np.float64)
            self.comm.Bcast(active_u, root=0)
        else:
            active_u = 2 * (np.random.uniform(size=(self.num_live_points, self.x_dim)) - 0.5)
        active_v = self.transform(active_u)

        if self.use_mpi:
            if self.mpi_rank == 0:
                chunks = [[] for _ in range(self.mpi_size)]
                for i, chunk in enumerate(active_v):
                    chunks[i % self.mpi_size].append(chunk)
            else:
                chunks = None
            data = self.comm.scatter(chunks, root=0)
            active_logl = self.loglike(data)
            recv_active_logl = self.comm.gather(active_logl, root=0)
            recv_active_logl = self.comm.bcast(recv_active_logl, root=0)
            active_logl = np.concatenate(recv_active_logl, axis=0)
        else:
            active_logl = self.loglike(active_v)

        saved_v = []  # Stored points for posterior results
        saved_logl = []
        saved_logwt = []
        h = 0.0  # Information, initially 0.
        logz = -1e300  # ln(Evidence Z), initially Z=0
        logvol = np.log(1.0 - np.exp(-1.0 / self.num_live_points))
        fraction_remain = 1.0
        ncall = self.num_live_points  # number of calls we already made
        first_time = True
        nb = self.mpi_size * mcmc_batch_size

        for it in range(0, max_iters):

            # Worst object in collection and its weight (= volume * likelihood)
            worst = np.argmin(active_logl)
            logwt = logvol + active_logl[worst]

            # Update evidence Z and information h.
            logz_new = np.logaddexp(logz, logwt)
            h = (np.exp(logwt - logz_new) * active_logl[worst] + np.exp(logz - logz_new) * (h + logz) - logz_new)
            logz = logz_new

            # Add worst object to samples.
            saved_v.append(np.array(active_v[worst]))
            saved_logwt.append(logwt)
            saved_logl.append(active_logl[worst])

            expected_vol = np.exp(-it / self.num_live_points)

            # The new likelihood constraint is that of the worst object.
            loglstar = active_logl[worst]

            if expected_vol > volume_switch:

                nc = 0
                # Simple rejection sampling over prior
                while True:
                    u = 2 * (np.random.uniform(size=(1, self.x_dim)) - 0.5)
                    v = self.transform(u)
                    logl = self.loglike(v)
                    nc += 1
                    if logl > loglstar:
                        break
                if self.use_mpi:
                    recv_samples = self.comm.gather(u, root=0)
                    recv_likes = self.comm.gather(logl, root=0)
                    recv_nc = self.comm.gather(nc, root=0)
                    recv_samples = self.comm.bcast(recv_samples, root=0)
                    recv_likes = self.comm.bcast(recv_likes, root=0)
                    recv_nc = self.comm.bcast(recv_nc, root=0)
                    samples = np.concatenate(recv_samples, axis=0)
                    likes = np.concatenate(recv_likes, axis=0)
                    ncall += sum(recv_nc)
                else:
                    samples = np.array(u)
                    likes = np.array(logl)
                    ncall += nc
                for ib in range(0, self.mpi_size):
                    active_u[worst] = samples[ib, :]
                    active_v[worst] = self.transform(active_u[worst])
                    active_logl[worst] = likes[ib]

                if it % log_interval == 0 and self.log:
                    self.logger.info(
                        'Step [%d] loglstar [%5.4f] max logl [%5.4f] logz [%5.4f] vol [%6.5f] ncalls [%d]]' %
                        (it, loglstar, np.max(active_logl), logz, expected_vol, ncall))

            else:

                # MCMC
                if first_time or it % update_interval == 0:
                    self.trainer.train(active_u, max_iters=train_iters, noise=noise)
                    if num_test_samples > 0:
                        # Test multiple chains from worst point to check mixing
                        init_x = np.concatenate(
                            [active_u[worst:worst + 1, :] for i in range(num_test_samples)])
                        logl = np.concatenate(
                            [active_logl[worst:worst + 1] for i in range(num_test_samples)])
                        test_samples, _, _, scale, _ = self.trainer.sample(
                            loglike=self.loglike, init_x=init_x, logl=logl, loglstar=loglstar,
                            transform=self.transform, mcmc_steps=test_mcmc_steps + test_mcmc_burn_in,
                            max_prior=1, alpha=alpha)
                        np.save(os.path.join(self.logs['chains'], 'test_samples.npy'), test_samples)
                        self._chain_stats(test_samples, mean=np.mean(active_u, axis=0), std=np.std(active_u, axis=0))
                    first_time = False

                accept = False
                while not accept:
                    if nb == self.mpi_size * mcmc_batch_size:
                        # Get a new batch of trial points
                        nb = 0
                        idx = np.random.randint(
                            low=0, high=self.num_live_points, size=mcmc_batch_size)
                        init_x = active_u[idx, :]
                        logl = active_logl[idx]
                        samples, likes, latent, scale, nc = self.trainer.sample(
                            loglike=self.loglike, init_x=init_x, logl=logl, loglstar=loglstar,
                            transform=self.transform, mcmc_steps=mcmc_steps + mcmc_burn_in,
                            max_prior=1, alpha=alpha)
                        if self.use_mpi:
                            recv_samples = self.comm.gather(samples, root=0)
                            recv_likes = self.comm.gather(likes, root=0)
                            recv_nc = self.comm.gather(nc, root=0)
                            recv_samples = self.comm.bcast(recv_samples, root=0)
                            recv_likes = self.comm.bcast(recv_likes, root=0)
                            recv_nc = self.comm.bcast(recv_nc, root=0)
                            samples = np.concatenate(recv_samples, axis=0)
                            likes = np.concatenate(recv_likes, axis=0)
                            ncall += sum(recv_nc)
                        else:
                            ncall += nc
                    for ib in range(nb, self.mpi_size * mcmc_batch_size):
                        nb += 1
                        if np.all(samples[ib, 0, :] != samples[ib, -1, :]) and likes[ib, -1] > loglstar:
                            active_u[worst] = samples[ib, -1, :]
                            active_v[worst] = self.transform(active_u[worst])
                            active_logl[worst] = likes[ib, -1]
                            accept = True
                            break

                if it % log_interval == 0 and self.log:
                    acceptance, ess, jump_distance = self._chain_stats(samples, mean=np.mean(active_u, axis=0),
                                                                       std=np.std(active_u, axis=0))
                    np.save(
                        os.path.join(
                            self.logs['extra'],
                            'active_%s.npy' %
                            it),
                        active_v)
                    self.logger.info(
                        'Step [%d] loglstar [%5.4f] maxlogl [%5.4f] logz [%5.4f] vol [%6.5f] ncalls [%d] scale [%5.4f]' %
                        (it, loglstar, np.max(active_logl), logz, expected_vol, ncall, scale))
                    with open(os.path.join(self.logs['results'], 'results.csv'), 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([it, acceptance, np.min(ess), np.max(
                            ess), jump_distance, scale, loglstar, logz, fraction_remain, ncall])
                    self._save_samples(np.array(saved_v), np.exp(np.array(saved_logwt) - logz), np.array(saved_logl))

            # Shrink interval
            logvol -= 1.0 / self.num_live_points
            logz_remain = np.max(active_logl) - it / self.num_live_points
            fraction_remain = np.logaddexp(logz, logz_remain) - logz

            if self.log:
                self.trainer.writer.add_scalar('logz', logz, it)

            # Stopping criterion
            if fraction_remain < dlogz:
                break

        logvol = -len(saved_v) / self.num_live_points - np.log(self.num_live_points)
        for i in range(self.num_live_points):
            logwt = logvol + active_logl[i]
            logz_new = np.logaddexp(logz, logwt)
            h = (np.exp(logwt - logz_new) * active_logl[i] + np.exp(logz - logz_new) * (h + logz) - logz_new)
            logz = logz_new
            saved_v.append(np.array(active_v[i]))
            saved_logwt.append(logwt)
            saved_logl.append(active_logl[i])

        if self.log:
            with open(os.path.join(self.logs['results'], 'final.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['niter', 'ncall', 'logz', 'logzerr', 'h'])
                writer.writerow([it + 1, ncall, logz, np.sqrt(h / self.num_live_points), h])
            self._save_samples(np.array(saved_v), np.exp(np.array(saved_logwt) - logz), np.array(saved_logl))

        print("niter: {:d}\n ncall: {:d}\n nsamples: {:d}\n logz: {:6.3f} +/- {:6.3f}\n h: {:6.3f}"
              .format(it + 1, ncall, len(np.array(saved_v)), logz, np.sqrt(h / self.num_live_points), h))

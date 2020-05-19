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
import json
import glob
import torch
import numpy as np

from nnest.mcmc import MCMCSampler


class NestedSampler(MCMCSampler):

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
                 base_dist=None,
                 scale='',
                 num_live_points=1000
                 ):

        super(NestedSampler, self).__init__(x_dim, loglike, transform=transform, append_run_num=append_run_num,
                                            run_num=run_num, hidden_dim=hidden_dim, num_slow=num_slow,
                                            num_derived=num_derived, batch_size=batch_size, flow=flow,
                                            num_blocks=num_blocks, num_layers=num_layers, log_dir=log_dir,
                                            use_gpu=use_gpu, base_dist=base_dist, scale=scale)

        self.num_live_points = num_live_points
        self.sampler = 'nested'

        if self.log:
            self.logger.info('Num live points [%d]' % self.num_live_points)

        if self.log:
            with open(os.path.join(self.logs['results'], 'results.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'acceptance', 'min_ess',
                                 'max_ess', 'jump_distance', 'scale', 'loglstar', 'logz', 'fraction_remain', 'ncall'])

    def run(
            self,
            method=None,
            mcmc_steps=0,
            mcmc_burn_in=0,
            mcmc_num_chains=10,
            max_iters=1000000,
            update_interval=None,
            log_interval=None,
            dlogz=0.5,
            train_iters=500,
            volume_switch=-1.0,
            step_size=0.0,
            jitter=-1.0,
            num_test_mcmc_samples=0,
            test_mcmc_steps=1000,
            test_mcmc_burn_in=0):

        if method is None:
            method = 'rejection_prior'

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

        if step_size == 0.0:
            step_size = 2 / self.x_dim ** 0.5

        if self.log:
            self.logger.info('MCMC steps [%d]' % mcmc_steps)
            self.logger.info('Initial scale [%5.4f]' % step_size)
            self.logger.info('Volume switch [%5.4f]' % volume_switch)

        it = 0
        if self.resume and not self.logs['created']:
            for f in glob.glob(os.path.join(self.logs['checkpoint'], 'checkpoint_*.txt')):
                if int(f.split('/checkpoint_')[1].split('.txt')[0]) > it:
                    it = int(f.split('/checkpoint_')[1].split('.txt')[0])

        if it > 0:

            self.logger.info('Using checkpoint [%d]' % it)

            with open(os.path.join(self.logs['checkpoint'], 'checkpoint_%s.txt' % it), 'r') as f:
                data = json.load(f)
                logz = data['logz']
                h = data['h']
                logvol = data['logvol']
                ncall = data['ncall']
                fraction_remain = data['fraction_remain']
                method = data['method']

            active_u = np.load(os.path.join(self.logs['checkpoint'], 'active_u_%s.npy' % it))
            active_v = self.transform(active_u)
            active_logl = np.load(os.path.join(self.logs['checkpoint'], 'active_logl_%s.npy' % it))
            active_derived = np.load(os.path.join(self.logs['checkpoint'], 'active_derived_%s.npy' % it))
            saved_v = np.load(os.path.join(self.logs['checkpoint'], 'saved_v.npy')).tolist()
            saved_logl = np.load(os.path.join(self.logs['checkpoint'], 'saved_logl.npy')).tolist()
            saved_logwt = np.load(os.path.join(self.logs['checkpoint'], 'saved_logwt.npy')).tolist()

        else:

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
                active_logl, active_derived = self.loglike(active_v)

            saved_v = []  # Stored points for posterior results
            saved_logl = []
            saved_logwt = []

            h = 0.0  # Information, initially 0.
            logz = -1e300  # ln(Evidence Z), initially Z=0
            logvol = np.log(1.0 - np.exp(-1.0 / self.num_live_points))
            fraction_remain = 1.0
            ncall = self.num_live_points  # number of calls we already made

        first_time = True
        nb = self.mpi_size * mcmc_num_chains
        rejection_sample = volume_switch < 1
        ncs = []

        for it in range(it, max_iters):

            # Worst object in collection and its weight (= volume * likelihood)
            worst = np.argmin(active_logl)
            logwt = logvol + active_logl[worst]

            # Update evidence Z and information h.
            logz_new = np.logaddexp(logz, logwt)
            h = (np.exp(logwt - logz_new) * active_logl[worst] + np.exp(logz - logz_new) * (h + logz) - logz_new)
            logz = logz_new

            # Add worst object to samples.

            if self.num_derived > 0:
                saved_v.append(np.concatenate((active_v[worst], active_derived[worst])))
            else:
                saved_v.append(np.array(active_v[worst], copy=True))
            saved_logwt.append(logwt)
            saved_logl.append(active_logl[worst])

            expected_vol = np.exp(-it / self.num_live_points)

            # The new likelihood constraint is that of the worst object.
            loglstar = active_logl[worst]

            # Train flow
            if first_time or it % update_interval == 0:
                self.trainer.train(active_u, max_iters=train_iters, jitter=jitter)

                if num_test_mcmc_samples > 0:
                    # Test multiple chains from worst point to check mixing
                    init_x = np.concatenate(
                        [active_u[worst:worst + 1, :] for i in range(num_test_mcmc_samples)])
                    test_samples, _, _, _, _, scale, _ = self.mcmc_sample(
                        init_x=init_x, loglstar=loglstar, mcmc_steps=test_mcmc_steps + test_mcmc_burn_in,
                        max_prior=1, step_size=step_size, dynamic_step_size=True)
                    np.save(os.path.join(self.logs['chains'], 'test_samples.npy'), test_samples)
                    self._chain_stats(test_samples, mean=np.mean(active_u, axis=0), std=np.std(active_u, axis=0))
                first_time = False

            if method == 'rejection_prior' or method == 'rejection_flow':

                # Rejection sampling

                nc = 0

                if method == 'rejection_prior':

                    # Simple rejection sampling over prior
                    while True:
                        u = 2 * (np.random.uniform(size=(1, self.x_dim)) - 0.5)
                        v = self.transform(u)
                        logl, der = self.loglike(v)
                        nc += 1
                        if logl > loglstar:
                            break

                else:

                    # Rejection sampling using flow
                    u, logl, nc = self.rejection_sample(self.loglike, loglstar, init_x=active_u, max_prior=1)

                ncs.append(nc)
                mean_calls = np.mean(ncs[-10:])

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
                    derived_samples = np.array(der)
                    likes = np.array(logl)
                    ncall += nc
                for ib in range(0, self.mpi_size):
                    active_u[worst] = samples[ib, :]
                    active_v[worst] = self.transform(active_u[worst])
                    active_logl[worst] = likes[ib]
                    if self.num_derived > 0:
                        active_derived[worst] = derived_samples[ib, :]

                if it % log_interval == 0 and self.log:
                    self.logger.info(
                        'Step [%d] loglstar [%5.4e] max logl [%5.4e] logz [%5.4e] vol [%6.5e] ncalls [%d] mean '
                        'calls [%5.4f]' % (it, loglstar, np.max(active_logl), logz, expected_vol, ncall, mean_calls))

                if expected_vol < volume_switch >= 0 or (volume_switch < 0 and mean_calls > mcmc_steps):
                    # Switch to MCMC if rejection sampling becomes inefficient
                    self.logger.info('Switching to MCMC sampling')
                    method = 'mcmc'

            elif method == 'mcmc':

                # MCMC sampling

                accept = False
                while not accept:
                    if nb == self.mpi_size * mcmc_num_chains:
                        # Get a new batch of trial points
                        nb = 0
                        idx = np.random.randint(
                            low=0, high=self.num_live_points, size=mcmc_num_chains)
                        init_x = active_u[idx, :]
                        samples, latent_samples, derived_samples, likes, scale, nc = self.mcmc_sample(
                            init_x=init_x, loglstar=loglstar, mcmc_steps=mcmc_steps + mcmc_burn_in,
                            max_prior=1, step_size=step_size, dynamic_step_size=True)
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
                    for ib in range(nb, self.mpi_size * mcmc_num_chains):
                        nb += 1
                        if np.all(samples[ib, 0, :] != samples[ib, -1, :]) and likes[ib, -1] > loglstar:
                            active_u[worst] = samples[ib, -1, :]
                            active_v[worst] = self.transform(active_u[worst])
                            active_logl[worst] = likes[ib, -1]
                            if self.num_derived > 0:
                                active_derived[worst] = derived_samples[ib, -1, :]
                            accept = True
                            break

                if it % log_interval == 0 and self.log:
                    acceptance, ess, jump_distance = self._chain_stats(samples, mean=np.mean(active_u, axis=0),
                                                                       std=np.std(active_u, axis=0))
                    self.logger.info(
                        'Step [%d] loglstar [%5.4e] maxlogl [%5.4e] logz [%5.4e] vol [%6.5e] ncalls [%d] '
                        'scale [%5.4f]' % (it, loglstar, np.max(active_logl), logz, expected_vol, ncall, scale))
                    with open(os.path.join(self.logs['results'], 'results.csv'), 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([it, acceptance, np.min(ess), np.max(ess),
                                         jump_distance, scale, loglstar, logz, fraction_remain, ncall])

            # Shrink interval
            logvol -= 1.0 / self.num_live_points
            logz_remain = np.max(active_logl) - it / self.num_live_points
            fraction_remain = np.logaddexp(logz, logz_remain) - logz

            if self.log:
                self.trainer.writer.add_scalar('logz', logz, it)

            self.samples = np.array(saved_v)
            self.weights = np.exp(np.array(saved_logwt) - logz)
            self.loglikes = -np.array(saved_logl)

            if it % log_interval == 0 and self.log:
                np.save(os.path.join(self.logs['checkpoint'], 'active_u_%s.npy' % it), active_u)
                np.save(os.path.join(self.logs['checkpoint'], 'active_logl_%s.npy' % it), active_logl)
                np.save(os.path.join(self.logs['checkpoint'], 'active_derived_%s.npy' % it), active_derived)
                np.save(os.path.join(self.logs['checkpoint'], 'saved_v.npy'), saved_v)
                np.save(os.path.join(self.logs['checkpoint'], 'saved_logl.npy'), saved_logl)
                np.save(os.path.join(self.logs['checkpoint'], 'saved_logwt.npy'), saved_logwt)
                with open(os.path.join(self.logs['checkpoint'], 'checkpoint_%s.txt' % it), 'w') as f:
                    json.dump({'logz': logz, 'h': h, 'logvol': logvol, 'ncall': ncall,
                               'fraction_remain': fraction_remain, 'method': method}, f)
                self._save_samples(self.samples, self.loglikes, weights=self.weights)

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

        self.logz = logz
        self.samples = np.array(saved_v)
        self.weights = np.exp(np.array(saved_logwt) - logz)
        self.loglikes = -np.array(saved_logl)

        if self.log:
            with open(os.path.join(self.logs['results'], 'final.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['niter', 'ncall', 'logz', 'logzerr', 'h'])
                writer.writerow([it + 1, ncall, logz, np.sqrt(h / self.num_live_points), h])
            self._save_samples(self.samples, self.loglikes, weights=self.weights)

        print("niter: {:d}\n ncall: {:d}\n nsamples: {:d}\n logz: {:6.3f} +/- {:6.3f}\n h: {:6.3f}"
              .format(it + 1, ncall, len(np.array(saved_v)), logz, np.sqrt(h / self.num_live_points), h))

    def rejection_sample(
            self,
            loglstar,
            init_x=None,
            max_prior=None,
            enlargement_factor=1.3,
            constant_efficiency_factor=None):

        self.trainer.netG.eval()

        if init_x is not None:
            z, log_det_J = self.trainer.netG(torch.from_numpy(init_x).float().to(self.trainer.device))
            # We want max det dx/dz to set envelope for rejection sampling
            m = torch.max(-log_det_J)
            z = z.detach().cpu().numpy()
            r = np.max(np.linalg.norm(z, axis=1))
        else:
            r = 1

        if constant_efficiency_factor is not None:
            enlargement_factor = (1 / constant_efficiency_factor) ** (1 / self.x_dim)

        nc = 0
        while True:
            if hasattr(self.trainer.netG.base_dist, 'usample'):
                z = self.trainer.netG.base_dist.usample(sample_shape=(1,)) * enlargement_factor
            else:
                z = np.random.randn(self.x_dim)
                z = enlargement_factor * r * z * np.random.rand() ** (1. / self.x_dim) / np.sqrt(np.sum(z ** 2))
                z = np.expand_dims(z, 0)
            x, log_det_J = self.trainer.netG(torch.from_numpy(z).float().to(self.trainer.device), mode='inverse')
            delta_log_det_J = (log_det_J - m).detach()
            log_ratio_1 = delta_log_det_J.squeeze(dim=1)
            x = x.detach().cpu().numpy()

            # Check not out of prior range
            if np.any(np.abs(x) > max_prior):
                continue

            # Check volume constraint
            rnd_u = torch.rand(log_ratio_1.shape, device=self.trainer.device)
            ratio = (log_ratio_1).exp().clamp(max=1)
            if rnd_u > ratio:
                continue

            logl = self.loglike(self.transform(x))
            idx = np.where(np.isfinite(logl) & (logl < loglstar))[0]
            log_ratio_1[idx] = -np.inf
            ratio = (log_ratio_1).exp().clamp(max=1)

            nc += 1
            if rnd_u < ratio:
                break

        return x, logl, nc

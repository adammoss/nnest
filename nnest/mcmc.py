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

import torch
import numpy as np
from getdist.mcsamples import MCSamples
from getdist.chains import chainFiles
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')

from nnest.sampler import Sampler
from nnest.utils.evaluation import acceptance_rate, effective_sample_size, mean_jump_distance
from nnest.utils.buffer import Buffer


class MCMCSampler(Sampler):

    def __init__(self,
                 x_dim,
                 loglike,
                 transform=None,
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
            transform:
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

        super(MCMCSampler, self).__init__(x_dim, loglike, transform=transform, append_run_num=append_run_num,
                                          run_num=run_num, hidden_dim=hidden_dim, num_slow=num_slow,
                                          num_derived=num_derived, batch_size=batch_size, flow=flow,
                                          num_blocks=num_blocks, num_layers=num_layers, log_dir=log_dir,
                                          use_gpu=use_gpu, base_dist=base_dist, scale=scale, trainer=trainer,
                                          prior=prior, transform_prior=transform_prior)

        self.sampler = 'mcmc'
        self.oversample_rate = oversample_rate if oversample_rate > 0 else self.num_fast / self.x_dim

    def mcmc_sample(
            self,
            mcmc_steps=20,
            step_size=1.0,
            dynamic_step_size=False,
            num_chains=1,
            init_x=None,
            loglstar=None,
            show_progress=False,
            out_chain=None,
            max_start_tries=100,
            output_interval=None,
            stats_interval=None):

        self.trainer.netG.eval()

        samples = []
        latent_samples = []
        derived_samples = []
        likes = []

        if init_x is not None:
            num_chains = init_x.shape[0]
            z, _ = self.trainer.netG(torch.from_numpy(init_x).float().to(self.trainer.device))
            z = z.detach()
            # Add the inverse version of x rather than init_x due to numerical precision
            x = self.trainer.get_samples(z)
            logl, derived = self.loglike(x)
        else:
            for i in range(max_start_tries):
                z = torch.randn(num_chains, self.trainer.z_dim, device=self.trainer.device)
                z = z.detach()
                x = self.trainer.get_samples(z)
                logl, derived = self.loglike(x)
                logl_prior = self.prior(x)
                if np.all(logl > -1e30) and np.all(logl_prior) > -1e30:
                    break
                if i == max_start_tries - 1:
                    raise Exception('Could not find starting value')

        samples.append(x)
        latent_samples.append(z.cpu().numpy())
        derived_samples.append(derived)
        likes.append(logl)

        iters = tqdm(range(1, mcmc_steps + 1)) if show_progress else range(1, mcmc_steps + 1)
        scale = step_size
        accept = 0
        reject = 0
        ncall = 0

        if out_chain is not None:
            if num_chains == 1:
                files = [open(out_chain + '.txt', 'w')]
            else:
                files = [open(out_chain + '_%s.txt' % (ib + 1), 'w') for ib in range(num_chains)]

        for it in iters:

            # Propose a move in latent space
            dz = torch.randn_like(z) * scale
            if self.num_slow > 0 and np.random.uniform() < self.oversample_rate:
                fast = True
                dz[:, 0:self.num_slow] = 0.0
            else:
                fast = False
            z_prime = z + dz

            # Jacobian is det d f^{-1} (z)/dz
            x, log_det_J = self.trainer.netG.inverse(z)
            x_prime, log_det_J_prime = self.trainer.netG.inverse(z_prime)
            x_prime = x_prime.detach().cpu().numpy()
            log_ratio_1 = (log_det_J_prime - log_det_J).detach()

            if self.prior is not None:
                logl_prior = self.prior(x_prime)
                log_ratio_1[np.where(logl_prior < -1e30)] = -np.inf

            # Check prior and Jacobian is accepted
            rnd_u = torch.rand(log_ratio_1.shape, device=self.trainer.device)
            ratio = log_ratio_1.exp().clamp(max=1)
            mask = (rnd_u < ratio).int()
            logl_prime = np.full(num_chains, logl)
            derived_prime = np.copy(derived)

            # Only evaluate likelihood if prior and Jacobian is accepted
            for idx, im in enumerate(mask):
                if im:
                    if not fast:
                        ncall += 1
                    lp, der = self.loglike(x_prime[idx])
                    if self.prior is not None:
                        lp += logl_prior[idx]
                    if loglstar is not None:
                        if np.isfinite(lp) and lp >= loglstar:
                            logl_prime[idx] = lp
                            derived_prime[idx] = der
                        else:
                            mask[idx] = 0
                    else:
                        if lp >= logl[idx]:
                            logl_prime[idx] = lp
                        elif rnd_u[idx].cpu().numpy() < np.clip(np.exp(lp - logl[idx]), 0, 1):
                            logl_prime[idx] = lp
                            derived_prime[idx] = der
                        else:
                            mask[idx] = 0

            if 2 * torch.sum(mask).cpu().numpy() > num_chains:
                accept += 1
            else:
                reject += 1

            if dynamic_step_size:
                if accept > reject:
                    scale *= np.exp(1. / accept)
                if accept < reject:
                    scale /= np.exp(1. / reject)

            m = mask[:, None].float()
            z = (z_prime * m + z * (1 - m)).detach()
            derived = derived_prime * m.cpu().numpy() + derived * (1 - m.cpu().numpy())

            m = mask
            logl = logl_prime * m.cpu().numpy() + logl * (1 - m.cpu().numpy())

            samples.append(self.trainer.get_samples(z))
            latent_samples.append(z.cpu().numpy())
            derived_samples.append(derived)
            likes.append(logl)

            if output_interval is not None and it % output_interval == 0 and out_chain is not None:
                self._save_samples(np.transpose(np.array(self.transform(samples)), axes=[1, 0, 2]),
                                   np.transpose(np.array(likes), axes=[1, 0]),
                                   derived_samples=np.transpose(np.array(derived_samples), axes=[1, 0, 2]))

            if stats_interval is not None and it % stats_interval == 0:
                self.logger.info('MCMC step [%d], ncalls [%d] ' % (it, ncall))
                self._chain_stats(np.transpose(np.array(samples), axes=[1, 0, 2]))

        # Transpose samples so shape is (chain_num, iteration, dim)
        samples = np.transpose(np.array(samples), axes=[1, 0, 2])
        latent_samples = np.transpose(np.array(latent_samples), axes=[1, 0, 2])
        derived_samples = np.transpose(np.array(derived_samples), axes=[1, 0, 2])
        likes = np.transpose(np.array(likes), axes=[1, 0])

        if out_chain is not None:
            for ib in range(num_chains):
                files[ib].close()

        return samples, latent_samples, derived_samples, likes, scale, ncall

    def _chain_stats(self, samples, mean=None, std=None):
        acceptance = acceptance_rate(samples)
        if mean is None:
            mean = np.mean(np.reshape(samples, (-1, samples.shape[2])), axis=0)
        if std is None:
            std = np.std(np.reshape(samples, (-1, samples.shape[2])), axis=0)
        ess = effective_sample_size(samples, mean, std)
        jump_distance = mean_jump_distance(samples)
        self.logger.info(
            'Acceptance [%5.4f] min ESS [%5.4f] max ESS [%5.4f] average jump distance [%5.4f]' %
            (acceptance, np.min(ess), np.max(ess), jump_distance))
        return acceptance, ess, jump_distance

    def _save_samples(self, samples, loglikes, weights=None, derived_samples=None, min_weight=1e-30, outfile='chain'):
        if weights is None:
            weights = np.ones_like(loglikes)
        if len(samples.shape) == 2:
            # Single chain
            with open(os.path.join(self.logs['chains'], outfile + '.txt'), 'w') as f:
                for i in range(samples.shape[0]):
                    f.write("%.5E " % max(weights[i], min_weight))
                    f.write("%.5E " % -loglikes[i])
                    f.write(" ".join(["%.5E" % v for v in samples[i, :]]))
                    if derived_samples is not None:
                        f.write(" ")
                        f.write(" ".join(["%.5E" % v for v in derived_samples[i, :]]))
                    f.write("\n")
        elif len(samples.shape) == 3:
            # Multiple chains
            for ib in range(samples.shape[0]):
                with open(os.path.join(self.logs['chains'], outfile + '_%s.txt' % (ib + 1)), 'w') as f:
                    for i in range(samples.shape[1]):
                        f.write("%.5E " % max(weights[ib, i], min_weight))
                        f.write("%.5E " % -loglikes[ib, i])
                        f.write(" ".join(["%.5E" % v for v in samples[ib, i, :]]))
                        if derived_samples is not None:
                            f.write(" ")
                            f.write(" ".join(["%.5E" % v for v in derived_samples[ib, i, :]]))
                        f.write("\n")

    def _init_samples(self, num_chains=10, mcmc_steps=500, burn_in=100, propose_scale=0.01, temperature=1.0):
        if self.sample_prior is not None:
            x_current = self.sample_prior(num_chains)
        else:
            raise ValueError('Prior does not have sample method')
        samples = [x_current]
        loglike_current, _ = self.loglike(x_current)
        loglike_current += self.prior(x_current)
        for i in range(burn_in + mcmc_steps):
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
        samples = np.array(samples)
        samples = samples[burn_in:, :, :]
        return samples.reshape(samples.shape[0] * samples.shape[1], samples.shape[2])

    def _read_samples(self, fileroot, match='', ignore_rows=0.3, thin=1):
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
            num_training_examples=1000,
            bootstrap_mcmc_steps=100,
            bootstrap_burn_in=20,
            bootstrap_iters=20,
            bootstrap_fileroot='',
            bootstrap_match='',
            mcmc_steps=5000,
            num_chains=5,
            step_size=0,
            ignore_rows=0.3,
            stats_interval=200,
            output_interval=100,
            init_samples=None,
            jitter=-1.0,
            l2norm_decay=0.5,
            max_iters=500):

        if step_size == 0.0:
            step_size = 1 / self.x_dim ** 0.5

        if self.log:
            self.logger.info('Alpha [%5.4f]' % (step_size))

        buffer = Buffer(max_size=num_training_examples)

        # Initial samples
        if init_samples is not None:
            samples = init_samples
        elif bootstrap_fileroot:
            samples = self._read_samples(bootstrap_fileroot, match=bootstrap_match, ignore_rows=ignore_rows)
        else:
            samples = self._init_samples()

        buffer.insert(samples[np.random.choice(samples.shape[0], num_training_examples, replace=False)])

        ncall = 0

        l2norm = 1.0

        for it in range(1, bootstrap_iters + 1):

            if it > 1:
                samples, latent_samples, derived_samples, likes, scale, nc = self.mcmc_sample(
                    mcmc_steps=bootstrap_burn_in + bootstrap_mcmc_steps, num_chains=num_chains,
                    step_size=step_size, stats_interval=bootstrap_mcmc_steps)
                samples = self.transform(samples)
                samples = samples[:, bootstrap_burn_in:, :]
                samples = samples.reshape((samples.shape[0] * samples.shape[1], samples.shape[2]))
                buffer.insert(samples[np.random.choice(samples.shape[0], int(samples.shape[0] / 10), replace=False)])
                ncall += nc

            training_samples = buffer()[:, :self.x_dim]
            mean = np.mean(training_samples, axis=0)
            std = np.std(training_samples, axis=0)
            # Normalise samples
            training_samples = (training_samples - mean) / std
            self.transform = lambda x: x * std + mean
            self.trainer.train(training_samples, jitter=jitter, l2norm=l2norm, max_iters=max_iters)

            l2norm *= l2norm_decay

            self.logger.info('Bootstrap step [%d], ncalls [%d] ' % (it, ncall))

        self.trainer.train(training_samples, jitter=jitter)
        samples, latent_samples, derived_samples, likes, scale, nc = self.mcmc_sample(
            mcmc_steps=mcmc_steps, num_chains=num_chains, step_size=step_size,
            out_chain=os.path.join(self.logs['chains'], 'chain'), stats_interval=stats_interval,
            output_interval=output_interval)
        samples = self.transform(samples)
        ncall += nc

        self.samples = samples
        self.latent_samples = latent_samples
        self.loglikes = -likes

        print("ncall: {:d}\n".format(ncall))

"""
.. module:: mcmc
   :synopsis: Sampler base class
.. moduleauthor:: Adam Moss <adam.moss@nottingham.ac.uk>
"""

from __future__ import print_function
from __future__ import division

import os
import json
import logging

import torch
import numpy as np
from tqdm import tqdm

from nnest.trainer import Trainer
from nnest.utils.evaluation import acceptance_rate, effective_sample_size, mean_jump_distance
from nnest.utils.logger import create_logger, make_run_dir


class Sampler(object):

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
                 resume=True,
                 use_gpu=False,
                 base_dist=None,
                 scale='',
                 trainer=None,
                 transform_prior=True,
                 ):

        self.x_dim = x_dim
        self.num_derived = num_derived
        self.num_params = x_dim + num_derived

        assert x_dim > num_slow
        self.num_slow = num_slow
        self.num_fast = x_dim - num_slow

        if transform is None:
            self.transform = lambda x: x
        else:
            def safe_transform(x):
                if isinstance(x, list):
                    x = np.array(x)
                if len(x.shape) == 1:
                    assert x.shape[0] == self.x_dim
                    x = np.expand_dims(x, 0)
                return transform(x)
            self.transform = safe_transform
        
        def safe_loglike(x):
            if isinstance(x, list):
                x = np.array(x)
            if len(x.shape) == 1:
                assert x.shape[0] == self.x_dim
                x = np.expand_dims(x, 0)
            # Note the flow works in terms of rescaled coordinates. Transform back to the
            # original co-ordinates here to evaluate the likelihood
            res = loglike(self.transform(x))
            if isinstance(res, tuple):
                logl, derived = res
            else:
                logl = res
                # Set derived shape to be (batch size, 0)
                derived = np.array([[] for _ in x])
            if len(logl.shape) == 0:
                logl = np.expand_dims(logl, 0)
            logl[np.logical_not(np.isfinite(logl))] = -1e100
            if len(derived.shape) == 1 or derived.shape[1] != self.num_derived:
                raise ValueError('Is the number of derived parameters correct and derived has the correct shape?')
            return logl, derived

        self.loglike = safe_loglike

        sample_prior = getattr(prior, "sample", None)
        if callable(sample_prior):
            self.sample_prior = sample_prior
        else:
            self.sample_prior = None

        if prior is None:
            self.prior = lambda x: 0
        else:
            def safe_prior(x):
                if isinstance(x, list):
                    x = np.array(x)
                if len(x.shape) == 1:
                    assert x.shape[0] == self.x_dim
                    x = np.expand_dims(x, 0)
                if transform_prior:
                    return np.array([prior(self.transform(x)) for x in x])
                else:
                    return np.array([prior(x) for x in x])
            self.prior = safe_prior

        self.use_mpi = False
        try:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.mpi_size = self.comm.Get_size()
            self.mpi_rank = self.comm.Get_rank()
            if self.mpi_size > 1:
                self.use_mpi = True
        except:
            self.mpi_size = 1
            self.mpi_rank = 0

        self.log = not self.use_mpi or (self.use_mpi and self.mpi_rank == 0)

        args = locals()
        args.update(vars(self))

        if self.log:
            self.logs = make_run_dir(log_dir, run_num, append_run_num=append_run_num)
            log_dir = self.logs['run_dir']
            self._save_params(args)
        else:
            log_dir = None

        self.resume = resume

        self.logger = create_logger(__name__, level=logging.INFO)

        if trainer is None:
            self.trainer = Trainer(
                x_dim,
                hidden_dim=hidden_dim,
                num_slow=num_slow,
                batch_size=batch_size,
                flow=flow,
                num_blocks=num_blocks,
                num_layers=num_layers,
                log_dir=log_dir,
                log=self.log,
                use_gpu=use_gpu,
                base_dist=base_dist,
                scale=scale)
        else:
            self.trainer = trainer

        if self.log:
            self.logger.info('Num base params [%d]' % (self.x_dim))
            self.logger.info('Num derived params [%d]' % (self.num_derived))
            self.logger.info('Total params [%d]' % (self.num_params))

    def _save_params(self, my_dict):
        my_dict = {k: str(v) for k, v in my_dict.items()}
        with open(os.path.join(self.logs['info'], 'params.txt'), 'w') as f:
            json.dump(my_dict, f, indent=4)

    def _mcmc_sample(
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
                z = self.trainer.netG.prior.sample((num_chains,)).to(self.trainer.device)
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

    def _rejection_sample(
            self,
            loglstar,
            init_x=None,
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
            if hasattr(self.trainer.netG.prior, 'usample'):
                z = self.trainer.netG.prior.usample(sample_shape=(1,)) * enlargement_factor
            else:
                z = np.random.randn(self.x_dim)
                z = enlargement_factor * r * z * np.random.rand() ** (1. / self.x_dim) / np.sqrt(np.sum(z ** 2))
                z = np.expand_dims(z, 0)
            x, log_det_J = self.trainer.netG.inverse(torch.from_numpy(z).float().to(self.trainer.device))
            log_ratio_1 = (log_det_J - m).detach()
            x = x.detach().cpu().numpy()

            if self.prior(x) < -1e30:
                continue

            # Check volume constraint
            rnd_u = torch.rand(log_ratio_1.shape, device=self.trainer.device)
            ratio = log_ratio_1.exp().clamp(max=1)
            if rnd_u > ratio:
                continue

            logl = self.loglike(x)
            idx = np.where(np.isfinite(logl) & (logl < loglstar))[0]
            log_ratio_1[idx] = -np.inf
            ratio = log_ratio_1.exp().clamp(max=1)

            nc += 1
            if rnd_u < ratio:
                break

        return x, logl, nc
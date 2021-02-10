"""
.. module:: sampler
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
import matplotlib.pyplot as plt
try:
    import emcee
except:
    pass

from nnest.trainer import Trainer
from nnest.utils.evaluation import acceptance_rate, effective_sample_size, mean_jump_distance, gelman_rubin_diagnostic
from nnest.utils.logger import create_logger, get_or_create_run_dir


class Sampler(object):

    def __init__(self,
                 x_dim,
                 loglike,
                 transform=None,
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
                 resume=True,
                 use_gpu=False,
                 base_dist=None,
                 scale='',
                 trainer=None,
                 transform_prior=True,
                 oversample_rate=-1,
                 log_level=logging.INFO,
                 param_names=None,
                 ):

        """
        Args:
            x_dim:
            loglike:
            transform:
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
            resume:
            use_gpu:
            base_dist:
            scale:
            trainer:
            transform_prior:
            log_level:
            param_names:
        """

        self.x_dim = x_dim
        self.num_derived = num_derived
        self.num_params = x_dim + num_derived

        assert x_dim > num_slow
        self.num_slow = num_slow
        self.num_fast = x_dim - num_slow

        self.param_names = param_names
        if self.param_names is not None:
            assert len(param_names) == self.num_params

        self.oversample_rate = oversample_rate if oversample_rate > 0 else self.num_fast / self.x_dim

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
            self.total_calls += x.shape[0]
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

        self.single_or_primary_process = not self.use_mpi or (self.use_mpi and self.mpi_rank == 0)

        args = locals()
        args.update(vars(self))

        if self.single_or_primary_process or os.path.isdir(os.path.join(log_dir, 'info')):
            self.logs = get_or_create_run_dir(log_dir, append_run_num=append_run_num)
            self.log_dir = self.logs['run_dir']
        else:
            self.logs = None
            self.log_dir = None

        if self.single_or_primary_process:
            self._save_params(args)

        self.resume = resume

        self.logger = create_logger(__name__, level=log_level)

        if trainer is None:
            self.trainer = Trainer(
                x_dim,
                hidden_dim=hidden_dim,
                num_slow=num_slow,
                batch_size=batch_size,
                flow=flow,
                num_blocks=num_blocks,
                num_layers=num_layers,
                learning_rate=learning_rate,
                log_dir=self.log_dir,
                log=self.single_or_primary_process,
                use_gpu=use_gpu,
                base_dist=base_dist,
                scale=scale)
        else:
            self.trainer = trainer

        if self.single_or_primary_process:
            self.logger.info('Num base params [%d]' % (self.x_dim))
            self.logger.info('Num derived params [%d]' % (self.num_derived))
            self.logger.info('Total params [%d]' % (self.num_params))

        self.total_accepted = 0
        self.total_rejected = 0
        self.total_calls = 0
        self.total_fast_calls = 0

    def _save_params(self, my_dict):
        my_dict = {k: str(v) for k, v in my_dict.items()}
        with open(os.path.join(self.logs['info'], 'params.txt'), 'w') as f:
            json.dump(my_dict, f, indent=4)

    def _mcmc_sample(
            self,
            mcmc_steps,
            step_size=0.0,
            dynamic_step_size=False,
            num_chains=1,
            init_samples=None,
            init_loglikes=None,
            init_derived=None,
            loglstar=None,
            show_progress=False,
            max_start_tries=100,
            output_interval=None,
            stats_interval=None,
            plot_trace=True):

        self.trainer.netG.eval()

        if step_size <= 0.0:
            step_size = 2 / self.x_dim ** 0.5

        samples = []
        latent_samples = []
        derived_samples = []
        loglikes = []

        iters = tqdm(range(1, mcmc_steps + 1)) if show_progress else range(1, mcmc_steps + 1)
        scale = step_size
        accept = 0
        reject = 0
        ncall = 0

        if init_samples is not None:
            num_chains = init_samples.shape[0]
            z, _ = self.trainer.forward(init_samples)
            # Add the inverse version of x rather than init_samples due to numerical precision
            x = self.trainer.get_samples(z, to_numpy=True)
            if init_loglikes is None or init_derived is None:
                logl, derived = self.loglike(x)
                ncall += num_chains
            else:
                logl = init_loglikes
                derived = init_derived
            logl_prior = self.prior(x)
        else:
            for i in range(max_start_tries):
                z = self.trainer.get_prior_samples(num_chains)
                x = self.trainer.get_samples(z, to_numpy=True)
                logl, derived = self.loglike(x)
                ncall += num_chains
                logl_prior = self.prior(x)
                if np.all(logl > -1e30) and np.all(logl_prior) > -1e30:
                    break
                if i == max_start_tries - 1:
                    raise Exception('Could not find starting value')

        samples.append(x)
        latent_samples.append(z.cpu().numpy())
        derived_samples.append(derived)
        loglikes.append(logl)

        for it in iters:

            self.logger.debug('z={}'.format(z))

            x, log_det_J = self.trainer.inverse(z)
            x = x.cpu().numpy()
            self.logger.debug('x={}'.format(x))

            if loglstar is not None:

                # Sampling from a hard likelihood constraint
                x_prime = x
                z_prime = z

                # Find a move that satisfies prior and Jacobian
                for i in range(5):

                    # Propose a move in latent space
                    dz = torch.randn_like(z) * scale
                    if self.num_slow > 0 and np.random.uniform() < self.oversample_rate:
                        fast = True
                        dz[:, 0:self.num_slow] = 0.0
                    else:
                        fast = False
                    z_propose = z + dz

                    self.logger.debug('z_propose={}'.format(z_propose))

                    try:
                        x_propose, log_det_J_propose = self.trainer.inverse(z_propose)
                    except ValueError:
                        #self.logger.error('Could not find inverse', z_propose)
                        continue
                    x_propose = x_propose.cpu().numpy()
                    log_ratio = log_det_J_propose - log_det_J

                    self.logger.debug('x_propose={}'.format(x_propose))

                    logl_prior = self.prior(x_propose)
                    log_ratio[np.where(logl_prior < -1e30)] = -np.inf

                    # Check prior and Jacobian is accepted
                    rnd_u = torch.rand(log_ratio.shape, device=self.trainer.device)
                    ratio = log_ratio.exp().clamp(max=1)
                    mask = (rnd_u < ratio).int()

                    self.logger.debug('Mask={}'.format(mask))

                    m = mask[:, None].float()
                    z_prime = (z_propose * m + z_prime * (1 - m)).detach()
                    x_prime = x_propose * m.cpu().numpy() + x_prime * (1 - m.cpu().numpy())

                self.logger.debug('z_prime={}'.format(z_prime))
                self.logger.debug('x_prime={}'.format(x_prime))

                self.logger.debug('Pre-likelihood mask={}'.format(mask))

                logl_prior_prime = self.prior(x_prime)
                derived_prime = np.copy(derived)
                # Only evaluate likelihood if prior and Jacobian is accepted
                logl_prime = np.full(num_chains, logl)
                for idx, im in enumerate(mask):
                    if im:
                        if fast:
                            self.total_fast_calls += 1
                        ncall += 1
                        lp, der = self.loglike(x_prime[idx])
                        if np.isfinite(lp) and lp >= loglstar:
                            logl_prime[idx] = lp
                            derived_prime[idx] = der
                        else:
                            mask[idx] = 0

                self.logger.debug('Post-likelihood mask={}'.format(mask))

            else:

                # Use likelihood and prior in proposal ratio

                # Propose a move in latent space
                dz = torch.randn_like(z) * scale
                if self.num_slow > 0 and np.random.uniform() < self.oversample_rate:
                    fast = True
                    dz[:, 0:self.num_slow] = 0.0
                else:
                    fast = False
                z_prime = z + dz

                self.logger.debug('z={}'.format(z))
                self.logger.debug('z_prime={}'.format(z_prime))

                try:
                    x_prime, log_det_J_prime = self.trainer.inverse(z_prime)
                except ValueError:
                    #self.logger.error('Could not find inverse', z_prime)
                    continue

                x_prime = x_prime.cpu().numpy()
                self.logger.debug('x_prime={}'.format(x_prime))

                ncall += num_chains
                if fast:
                    self.total_fast_calls += num_chains
                logl_prime, derived_prime = self.loglike(x_prime)
                logl_prior_prime = self.prior(x_prime)
                log_ratio_1 = log_det_J_prime - log_det_J
                log_ratio_2 = torch.tensor(logl_prime - logl)
                log_ratio_3 = torch.tensor(logl_prior_prime - logl_prior)
                log_ratio = log_ratio_1 + log_ratio_2 + log_ratio_3

                self.logger.debug('log ratio 1={}'.format(log_ratio_1))
                self.logger.debug('log ratio 2={}'.format(log_ratio_2))
                self.logger.debug('log ratio 3={}'.format(log_ratio_3))
                self.logger.debug('log ratio={}'.format(log_ratio))

                rnd_u = torch.rand(log_ratio.shape, device=self.trainer.device)
                ratio = log_ratio.exp().clamp(max=1)
                mask = (rnd_u < ratio).int()

                self.logger.debug('Mask={}'.format(mask))

            num_accepted = torch.sum(mask).cpu().numpy()
            self.total_accepted += num_accepted
            self.total_rejected += num_chains - num_accepted

            if dynamic_step_size:
                if 2 * num_accepted > num_chains:
                    accept += 1
                else:
                    reject += 1
                if accept > reject:
                    scale *= np.exp(1. / (1 + accept))
                if accept < reject:
                    scale /= np.exp(1. / (1 + reject))
                self.logger.debug('scale=%5.4f accept=%d reject=%d' % (scale, accept, reject))

            logl = logl_prime * mask.cpu().numpy() + logl * (1 - mask.cpu().numpy())
            # Avoid multiplying due to -np.inf
            logl_prior[np.where(mask.cpu().numpy() == 1)] = logl_prior_prime[np.where(mask.cpu().numpy() == 1)]
            m = mask[:, None].float()
            z = (z_prime * m + z * (1 - m)).detach()
            x = x_prime * m.cpu().numpy() + x * (1 - m.cpu().numpy())
            derived = derived_prime * m.cpu().numpy() + derived * (1 - m.cpu().numpy())

            samples.append(x)
            latent_samples.append(z.cpu().numpy())
            derived_samples.append(derived)
            loglikes.append(logl)

            if output_interval is not None and it % output_interval == 0:
                self._save_samples(np.transpose(np.array(self.transform(samples)), axes=[1, 0, 2]),
                                   np.transpose(np.array(loglikes), axes=[1, 0]),
                                   derived_samples=np.transpose(np.array(derived_samples), axes=[1, 0, 2]))

            if stats_interval is not None and it % stats_interval == 0:
                self._chain_stats(np.transpose(np.array(self.transform(samples)), axes=[1, 0, 2]), step=it)

        # Transpose samples so shape is (chain_num, iteration, dim)
        samples = np.transpose(np.array(samples), axes=[1, 0, 2])
        latent_samples = np.transpose(np.array(latent_samples), axes=[1, 0, 2])
        derived_samples = np.transpose(np.array(derived_samples), axes=[1, 0, 2])
        loglikes = np.transpose(np.array(loglikes), axes=[1, 0])

        if plot_trace:
            self._plot_trace(samples, latent_samples)

        return samples, latent_samples, derived_samples, loglikes, scale, ncall

    def _plot_trace(self, samples, latent_samples):
        if self.log_dir is not None:
            fig, ax = plt.subplots(self.x_dim, 2, figsize=(10, self.x_dim), sharex=True)
            for i in range(self.x_dim):
                ax[i, 0].plot(samples[0, :, i])
                ax[i, 1].plot(latent_samples[0, 0:1000, i])
            plt.savefig(os.path.join(self.log_dir, 'plots', 'trace.png'))
            plt.close()

    def _chain_stats(self, samples, mean=None, std=None, step=None):
        acceptance = acceptance_rate(samples)
        if mean is None:
            mean = np.mean(np.reshape(samples, (-1, samples.shape[2])), axis=0)
        if std is None:
            std = np.std(np.reshape(samples, (-1, samples.shape[2])), axis=0)
        ess = effective_sample_size(samples, mean, std)
        jump_distance = mean_jump_distance(samples)
        grd = gelman_rubin_diagnostic(samples)
        if step is None:
            self.logger.info(
                'Acceptance [%5.4f] min ESS [%5.4f] max ESS [%5.4f] average jump [%5.4f]' %
                (acceptance, np.min(ess), np.max(ess), jump_distance))
        else:
            self.logger.info(
                'Step [%d] acceptance [%5.4f] min ESS [%5.4f] max ESS [%5.4f] average jump [%5.4f]' %
                (step, acceptance, np.min(ess), np.max(ess), jump_distance))
        return acceptance, ess, jump_distance

    def _save_samples(self, samples, loglikes, weights=None, derived_samples=None, min_weight=1e-30, outfile='chain'):
        if weights is None:
            weights = np.ones_like(loglikes)
        if len(samples.shape) == 2:
            # Single chain
            with open(os.path.join(self.logs['chains'], outfile + '.txt'), 'w') as f:
                if self.param_names is not None:
                    f.write("#weight minusloglike ")
                    f.write(" ".join(self.param_names))
                    f.write("\n")
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
                    if self.param_names is not None:
                        f.write("#weight minusloglike ")
                        f.write(" ".join(self.param_names))
                        f.write("\n")
                    for i in range(samples.shape[1]):
                        f.write("%.5E " % max(weights[ib, i], min_weight))
                        f.write("%.5E " % -loglikes[ib, i])
                        f.write(" ".join(["%.5E" % v for v in samples[ib, i, :]]))
                        if derived_samples is not None:
                            f.write(" ")
                            f.write(" ".join(["%.5E" % v for v in derived_samples[ib, i, :]]))
                        f.write("\n")

    def _rejection_prior_sample(self, loglstar):

        ncall = 0
        while True:
            x = self.sample_prior(1)
            logl, derived = self.loglike(x)
            ncall += 1
            if logl > loglstar:
                break

        return x, logl, derived, ncall

    def _rejection_flow_sample(
            self,
            init_samples,
            loglstar,
            enlargement_factor=1.1,
            constant_efficiency_factor=None,
            cache=False):

        self.trainer.netG.eval()

        def get_cache():
            z, log_det_J = self.trainer.forward(init_samples)
            # We want max det dx/dz to set envelope for rejection sampling
            self.max_log_det_J = enlargement_factor * torch.max(-log_det_J)
            z = z.cpu().numpy()
            self.max_r = np.max(np.linalg.norm(z, axis=1))

        if not cache:
            get_cache()
        else:
            try:
                self.max_log_det_J
            except:
                get_cache()

        if constant_efficiency_factor is not None:
            enlargement_factor = (1 / constant_efficiency_factor) ** (1 / self.x_dim)

        ncall = 0
        while True:
            if hasattr(self.trainer.netG.prior, 'usample'):
                z = self.trainer.netG.prior.usample(sample_shape=(1,)) * enlargement_factor
            else:
                z = np.random.randn(self.x_dim)
                z = enlargement_factor * self.max_r * z * np.random.rand() ** (1. / self.x_dim) / np.sqrt(
                    np.sum(z ** 2))
                z = np.expand_dims(z, 0)
            try:
                x, log_det_J = self.trainer.inverse(z)
            except ValueError:
                #self.logger.error('Could not find inverse', z)
                continue

            x = x.cpu().numpy()
            if self.prior(x) < -1e30:
                continue

            # Check Jacobian constraint
            log_ratio = log_det_J - self.max_log_det_J
            rnd_u = torch.rand(log_ratio.shape)
            ratio = log_ratio.exp().clamp(max=1)
            if rnd_u > ratio:
                continue

            logl, derived = self.loglike(x)
            idx = np.where(np.isfinite(logl) & (logl < loglstar))[0]
            log_ratio[idx] = -np.inf
            ratio = log_ratio.exp().clamp(max=1)
            ncall += 1
            if rnd_u < ratio:
                break

        return x, logl, derived, ncall

    def _density_sample(self, loglstar):

        self.trainer.netG.eval()

        ncall = 0
        while True:

            z = self.trainer.get_prior_samples(1)
            try:
                x = self.trainer.get_samples(z, to_numpy=True)
            except:
                continue

            if self.prior(x) < -1e30:
                continue

            logl, derived = self.loglike(x)
            ncall += 1
            if logl[0] > loglstar:
                break

        return x, logl, derived, ncall

    def _ensemble_sample(
            self,
            mcmc_steps,
            num_walkers,
            init_samples=None,
            init_loglikes=None,
            init_derived=None,
            loglstar=None,
            show_progress=False,
            max_start_tries=100,
            output_interval=None,
            stats_interval=None,
            plot_trace=True,
            moves=None):

        self.trainer.netG.eval()

        samples = []
        latent_samples = []
        derived_samples = []
        loglikes = []

        iters = tqdm(range(1, mcmc_steps + 1)) if show_progress else range(1, mcmc_steps + 1)

        if init_samples is not None:
            if isinstance(init_samples, emcee.State):
                state = emcee.State(init_samples)
            else:
                num_walkers = init_samples.shape[0]
                z, _ = self.trainer.forward(init_samples, to_numpy=True)
                state = emcee.State(z, log_prob=init_loglikes, blobs=init_derived)
        else:
            for i in range(max_start_tries):
                z = self.trainer.get_prior_samples(num_walkers, to_numpy=True)
                x = self.trainer.get_samples(z, to_numpy=True)
                logl_prior = self.prior(x)
                if np.all(logl_prior) > -1e30:
                    break
                if i == max_start_tries - 1:
                    raise Exception('Could not find starting value')
            state = emcee.State(z)

        def transformed_loglike(z):
            assert z.shape == (self.x_dim,), z.shape
            try:
                x, log_det_J = self.trainer.inverse(z.reshape((1, -1)), to_numpy=True)
            except:
                return -np.inf, np.zeros((1, self.num_derived))
            logl, der = self.loglike(x)
            if loglstar is not None:
                if logl < loglstar:
                    return -np.inf, der
                else:
                    return log_det_J + self.prior(x), der
            else:
                return logl + log_det_J + self.prior(x), np.zeros((1, self.num_derived))

        sampler = emcee.EnsembleSampler(num_walkers, self.x_dim, transformed_loglike, moves=moves)

        ncall = num_walkers if init_loglikes is None else 0

        for it in iters:

            state = sampler.run_mcmc(state, 1)
            z = state.coords
            derived = state.blobs

            ncall += num_walkers
            x, log_det_J = self.trainer.inverse(z, to_numpy=True)

            samples.append(x)
            latent_samples.append(z)
            derived_samples.append(derived)
            loglikes.append(state.log_prob)

            if output_interval is not None and it % output_interval == 0:
                self._save_samples(np.transpose(np.array(self.transform(samples)), axes=[1, 0, 2]),
                                   np.transpose(np.array(loglikes), axes=[1, 0]),
                                   derived_samples=np.transpose(np.array(derived_samples), axes=[1, 0, 2]))

            if stats_interval is not None and it % stats_interval == 0 and it > 1:
                self._chain_stats(np.transpose(np.array(self.transform(samples)), axes=[1, 0, 2]), step=it)

        # Transpose samples so shape is (chain_num, iteration, dim)
        samples = np.transpose(np.array(samples), axes=[1, 0, 2])
        latent_samples = np.transpose(np.array(latent_samples), axes=[1, 0, 2])
        derived_samples = np.transpose(np.array(derived_samples), axes=[1, 0, 2])
        loglikes = np.transpose(np.array(loglikes), axes=[1, 0])

        if plot_trace:
            self._plot_trace(samples, latent_samples)

        return samples, latent_samples, derived_samples, loglikes, ncall

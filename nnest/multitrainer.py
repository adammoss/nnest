"""
.. module:: trainer
   :synopsis: Train the flow and perform sampling in latent space
.. moduleauthor:: Adam Moss <adam.moss@nottingham.ac.uk>
"""

from __future__ import print_function
from __future__ import division

import os, sys
import time

import torch
from torch.utils.data import DataLoader
import numpy as np
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
import scipy.spatial
from tqdm import tqdm

import emcee

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nnest.networks import SingleSpeed, FastSlow, BatchNormFlow
from nnest.utils.logger import create_logger
from nnest.trainer import Trainer


class MultiTrainer(Trainer):

    def __init__(self,
                 xdim,
                 ndim,
                 nslow=0,
                 batch_size=100,
                 flow='nvp',
                 num_blocks=5,
                 num_layers=2,
                 oversample_rate=-1,
                 train=True,
                 load_model='',
                 log_dir='logs',
                 use_gpu=False,
                 log=True
                 ):

        self.device = torch.device(
            'cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

        self.x_dim = xdim
        self.z_dim = xdim

        self.batch_size = batch_size
        self.total_iters = 0

        assert xdim > nslow
        nfast = xdim - nslow
        self.nslow = nslow

        if oversample_rate > 0:
            self.oversample_rate = oversample_rate
        else:
            self.oversample_rate = nfast / xdim
        
        def init_network():
            if flow.lower() == 'nvp':
                if nslow > 0:
                    self.netG = FastSlow(nfast, nslow, ndim, num_blocks, num_layers)
                else:
                    self.netG = SingleSpeed(xdim, ndim, num_blocks, num_layers)
            else:
                raise NotImplementedError
            self.nparams = sum(p.numel() for p in self.netG.parameters())

            if train and not load_model:
                if log_dir is not None:
                    self.path = log_dir
                    if not os.path.exists(os.path.join(self.path, 'models')):
                        os.makedirs(os.path.join(self.path, 'models'))
                    if not os.path.exists(os.path.join(self.path, 'data')):
                        os.makedirs(os.path.join(self.path, 'data'))
                    if not os.path.exists(os.path.join(self.path, 'chains')):
                        os.makedirs(os.path.join(self.path, 'chains'))
                    if not os.path.exists(os.path.join(self.path, 'plots')):
                        os.makedirs(os.path.join(self.path, 'plots'))
                else:
                    self.path = None
            else:
                self.path = os.path.join(log_dir, load_model)
                self.netG.load_state_dict(torch.load(
                    os.path.join(self.path, 'models', 'netG.pt')
                ))
            self.optimizer = torch.optim.Adam(
                self.netG.parameters(), lr=0.0001, weight_decay=1e-6)

            if self.path is not None:
                self.writer = SummaryWriter(self.path)
        
        self.init_network = init_network
        self.init_network()
        print(self.netG)

        self.logger = create_logger(__name__)
        self.log = log

    def train(
            self,
            samples,
            max_iters=5000,
            log_interval=50,
            save_interval=50,
            noise=0.0,
            validation_fraction=0.05):
        
        assert samples.shape[1] == self.x_dim, samples.shape

        start_time = time.time()

        if self.path:
            fig, ax = plt.subplots()
            ax.scatter(samples[:, 0], samples[:, 1])
            self.writer.add_figure('originals', fig, self.total_iters)
            np.save(
                os.path.join(self.path, 'data', 'originals.npy'),
                samples)

        if noise < 0:
            # compute distance to nearest neighbor
            kdt = scipy.spatial.cKDTree(samples)
            dists, neighs = kdt.query(samples, 2)
            training_noise = 0.5 * np.mean(dists) / self.x_dim
        else:
            training_noise = noise

        if self.log:
            self.logger.info('Number of training samples [%d] for [%d] variables' % (samples.shape[0], self.nparams))
            self.logger.info('Training noise [%5.4f]' % training_noise)

        X_train, X_valid = train_test_split(
            samples, test_size=validation_fraction)

        train_tensor = torch.from_numpy(X_train.astype(np.float32))
        train_dataset = torch.utils.data.TensorDataset(train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)

        valid_tensor = torch.from_numpy(X_valid.astype(np.float32))
        valid_dataset = torch.utils.data.TensorDataset(valid_tensor)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=X_valid.shape[0], shuffle=False, drop_last=True)

        best_validation_loss = float('inf')
        best_validation_epoch = 0
        best_model = self.netG.state_dict()
        self.netGs = []
        self.netGs_accepts = []
        self.netGs_samples = []
        self.netGs_init_x = []

        stored = '  '
        for epoch in range(1, max_iters + 1):

            self.total_iters += 1

            train_loss = self._train(
                epoch, self.netG, train_loader, noise=training_noise)
            validation_loss = self._validate(epoch, self.netG, valid_loader)
            
            # 1) remember only if much better
            # 2) when performance is improving drastically at each epoch, avoid storing every epoch.
            if validation_loss < best_validation_loss - 1.0 and epoch >= best_validation_epoch + save_interval:
                best_validation_epoch = epoch
                best_validation_loss = validation_loss
                best_model = self.netG.state_dict()
                self.netGs.append(self.netG.state_dict())
                self.netGs_accepts.append(0)
                self.netGs_samples.append(0)
                self.netGs_init_x.append(None)
                stored = '**'

            if epoch == 1 or epoch % log_interval == 0:
                print(
                    'Epoch: {} validation loss: {:6.4f} {}'.format(
                        epoch, validation_loss, stored))
                stored = '  '
                
            #if epoch % save_interval == 0:

            if self.path:
                self.writer.add_scalar('loss', validation_loss, self.total_iters)
                if epoch % save_interval == 0:
                    torch.save(
                        self.netG.state_dict(),
                        os.path.join(self.path, 'models', 'netG.pt%d' % epoch)
                    )
                    if self.x_dim == 2:
                        self._train_plot(self.netG, samples)
        self.netG.load_state_dict(best_model)
    
    def choose_netG(self, verbose=False):
        # compute accept probabilities
        # give initially equal acceptance probabilities
        if verbose:
            print("Sampling from networks:", ' '.join(['%4d/%4d' % (A, S) for A, S in zip(self.netGs_accepts, self.netGs_samples)]), end=" \r")
            sys.stdout.flush()
        probs = np.array([(A + 1.) / (S + 1) for A, S in zip(self.netGs_accepts, self.netGs_samples)])
        probs /= probs.sum()
        i = np.random.choice(np.arange(len(probs)), p=probs)
        self.netG.load_state_dict(self.netGs[i])
        #print("Network %d: %d/%d" % (i, self.netGs_accepts[i], self.netGs_samples[i]))
        return i

    def sample(
            self,
            mcmc_steps=20,
            alpha=1.0,
            dynamic=True,
            batch_size=1,
            loglike=None,
            init_x=None,
            logl=None,
            loglstar=None,
            transform=None,
            show_progress=False,
            plot=False,
            out_chain=None,
            max_prior=None,
            nwalkers=40):
        
        def transformed_logpost(z):
            assert z.shape == (self.x_dim,), z.shape
            x, log_det_J = self.netG(torch.from_numpy(z.reshape((1,-1))).float().to(self.device), mode='inverse')
            x = x.detach().cpu().numpy()
            lnlike = float(loglike(transform(x)))
            log_det_J = float(log_det_J.detach())
            logprob = log_det_J + lnlike
            #print('Like=%.1f' % logprob, x)
            return logprob

        allsamples = []
        alllatent = []
        alllikes = []
        allncall = 0
        populations = [None for net in self.netGs]
        nsteps = 4
        #nsamples = 200 // len(self.netGs_init_x)
        while allncall < mcmc_steps:
            neti = self.choose_netG(verbose = True)
            sampler = populations[neti]
            if sampler is None:
                z0 = np.random.normal(size=(nwalkers, self.x_dim))
                if init_x is not None:
                    batch_size = init_x.shape[0]
                    z, _ = self.netG(torch.from_numpy(init_x).float().to(self.device))
                    z = z.detach()
                    z0[:batch_size,:] = z
                
                sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=self.x_dim, lnpostfn=transformed_logpost)
                pos, lnprob, rstate = sampler.run_mcmc(z0, nsteps)
                populations[neti] = sampler
            else:
                # advance population
                pos, lnprob, rstate = sampler.run_mcmc(sampler.chain[:,-1,:], lnprob0=sampler.lnprobability[:,-1], N=nsteps)
            
            alllatent.append(pos)
            alllikes.append(lnprob)
            allncall += nsteps * nwalkers
            x, log_det_J = self.netG(torch.from_numpy(pos).float().to(self.device), mode='inverse')
            x = x.detach().cpu().numpy()
            allsamples.append(x)
            
            ar = sampler.acceptance_fraction
            Nsamples = len(sampler.flatchain)
            #self.logger.info('Network %d sampler accepted: %.3f (%.d/%d)' % (neti, ar.mean(), ar.mean() * Nsamples, Nsamples))
            self.netGs_accepts[neti] = int(ar.mean() * Nsamples)
            self.netGs_samples[neti] = Nsamples
        
        #print(np.shape(allsamples), np.shape(alllikes), np.shape(alllatent))
        # Transpose so shape is (chain_num, iteration, dim)
        #samples = np.transpose(np.array(allsamples), axes=[1, 0, 2])
        #latent = np.transpose(np.array(alllatent), axes=[1, 0, 2])
        #likes = np.transpose(np.array(alllikes), axes=[1, 0])
        samples = np.array(allsamples).reshape((1, -1, self.x_dim))
        latent = np.array(alllatent).reshape((1, -1, self.x_dim))
        likes = np.array(alllikes).reshape((1, -1))
        print()
        ncall = allncall

        if self.path and plot:
            cmap = plt.cm.jet
            cmap.set_under('w', 1)
            fig, ax = plt.subplots()
            ax.hist2d(samples[0, :, 0], samples[0, :, 1],
                      bins=200, cmap=cmap, vmin=1, alpha=0.2)
            #if self.writer is not None:
            #    self.writer.add_figure('chain', fig, self.total_iters)
            plt.tight_layout()
            plt.savefig(os.path.join(self.path, 'plots', 'chain_%s.png' % self.total_iters))
            plt.close()
            fig, ax = plt.subplots()
            ax.plot(likes[0, len(likes[0])//3:])
            self.logger.info('lnLike: %d+-%d -> %d+-%d' % (
                alllikes[len(alllikes)//3].mean(), alllikes[len(alllikes)//3].std(), 
                alllikes[-1].mean(), alllikes[-1].std()
            ))
            #if self.writer is not None:
            #    self.writer.add_figure('likeevol', fig, self.total_iters)
            plt.tight_layout()
            plt.savefig(os.path.join(self.path, 'plots', 'likeevol_%s.png' % self.total_iters))
            plt.close()

        return samples, likes, latent, alpha, ncall
        

    def subsample(
            self,
            mcmc_steps=20,
            alpha=1.0,
            dynamic=True,
            batch_size=1,
            loglike=None,
            init_x=None,
            logl=None,
            loglstar=None,
            transform=None,
            show_progress=False,
            plot=False,
            out_chain=None,
            max_prior=None,
            max_start_tries=100):

        self.netG.eval()

        samples = []
        latent = []
        likes = []

        if transform is None:
            def transform(x): return x

        if init_x is not None:
            batch_size = init_x.shape[0]
            z, _ = self.netG(torch.from_numpy(init_x).float().to(self.device))
            z = z.detach()
            # Add the backward version of x rather than init_x due to numerical precision
            x, _ = self.netG(z, mode='inverse')
            x = x.detach().cpu().numpy()
            if logl is None:
                logl = loglike(transform(x))
        else:
            if logl is None:
                for i in range(max_start_tries):
                    z = torch.randn(batch_size, self.z_dim, device=self.device)
                    x, _ = self.netG(z, mode='inverse')
                    x = x.detach().cpu().numpy()
                    logl = loglike(transform(x))
                    if np.all(logl > -1e30):
                        break
                    if i == max_start_tries - 1:
                        raise Exception('Could not find starting value')
            else:
                z = torch.randn(batch_size, self.z_dim, device=self.device)
                x, _ = self.netG(z, mode='inverse')
                x = x.detach().cpu().numpy()
                logl = loglike(transform(x))                   

        samples.append(x)
        likes.append(logl)

        iters = range(mcmc_steps)
        if show_progress:
            iters = tqdm(iters)

        scale = alpha
        accept = 0
        reject = 0
        ncall = 0

        if out_chain is not None:
            if batch_size == 1:
                files = [open(out_chain + '.txt', 'w')]
            else:
                files = [open(out_chain + '_%s.txt' % (ib + 1), 'w') for ib in range(batch_size)]

        for i in iters:

            dz = torch.randn_like(z) * scale
            if self.nslow > 0 and np.random.uniform() < self.oversample_rate:
                fast = True
                dz[:, 0:self.nslow] = 0.0
            else:
                fast = False
            z_prime = z + dz

            # Jacobian is det d f^{-1} (z)/dz
            x, log_det_J = self.netG(z, mode='inverse')
            x_prime, log_det_J_prime = self.netG(z_prime, mode='inverse')
            x = x.detach().cpu().numpy()
            x_prime = x_prime.detach().cpu().numpy()
            delta_log_det_J = (log_det_J_prime - log_det_J).detach()
            log_ratio_1 = delta_log_det_J.squeeze(dim=1)

            # Check not out of prior range
            if max_prior is not None:
                prior = np.logical_or(
                    np.abs(x) > max_prior,
                    np.abs(x_prime) > max_prior)
                idx = np.where([np.any(p) for p in prior])
                log_ratio_1[idx] = -np.inf

            rnd_u = torch.rand(log_ratio_1.shape, device=self.device)
            ratio = log_ratio_1.exp().clamp(max=1)
            mask = (rnd_u < ratio).int()
            logl_prime = np.full(batch_size, logl)

            # Only evaluate likelihood if prior and volume is accepted
            if loglike is not None and transform is not None:
                for idx, im in enumerate(mask):
                    if im:
                        if not fast:
                            ncall += 1
                        lp = loglike(transform(x_prime[idx]))
                        if loglstar is not None:
                            if np.isfinite(lp) and lp >= loglstar:
                                logl_prime[idx] = lp
                            else:
                                mask[idx] = 0
                        else:
                            if lp >= logl[idx]:
                                logl_prime[idx] = lp
                            elif rnd_u[idx].cpu().numpy() < np.clip(np.exp(lp - logl[idx]), 0, 1):
                                logl_prime[idx] = lp
                            else:
                                mask[idx] = 0

            accept += torch.sum(mask).cpu().numpy()
            reject += batch_size - torch.sum(mask).cpu().numpy()

            if dynamic:
                if accept > reject:
                    scale *= np.exp(1. / accept)
                if accept < reject:
                    scale /= np.exp(1. / reject)

            m = mask[:, None].float()
            z = (z_prime * m + z * (1 - m)).detach()

            m = mask
            logl = logl_prime * m.cpu().numpy() + logl * (1 - m.cpu().numpy())

            x, _ = self.netG(z, mode='inverse')
            x = x.detach().cpu().numpy()
            samples.append(x)
            likes.append(logl)
            latent.append(z.cpu().numpy())

            if out_chain is not None:
                v = transform(x)
                for ib in range(batch_size):
                    files[ib].write("%.5E " % 1)
                    files[ib].write("%.5E " % -logl[ib])
                    files[ib].write(" ".join(["%.5E" % vi for vi in v[ib]]))
                    files[ib].write("\n")
                    files[ib].flush()
        
        if out_chain is not None:
            for ib in range(batch_size):
                files[ib].close()


        return samples, likes, latent, scale, ncall, accept


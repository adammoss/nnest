"""
.. module:: trainer
   :synopsis: Train the flow and perform sampling in latent space
.. moduleauthor:: Adam Moss <adam.moss@nottingham.ac.uk>
"""

from __future__ import print_function
from __future__ import division

import os
import time
import copy

import torch
from torch.utils.data import DataLoader
import numpy as np
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
import scipy.spatial
from tqdm import tqdm

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nnest.networks import SingleSpeed, FastSlow, BatchNormFlow
from nnest.utils.logger import create_logger


class Trainer(object):

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

        if flow.lower() == 'nvp':
            if nslow > 0:
                self.netG = FastSlow(nfast, nslow, ndim, num_blocks, num_layers, device=self.device)
            else:
                self.netG = SingleSpeed(xdim, ndim, num_blocks, num_layers, device=self.device)
        else:
            raise NotImplementedError

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

        self.logger = create_logger(__name__)
        self.log = log

        if self.path is not None:
            self.logger.info(self.netG)
            self.writer = SummaryWriter(self.path)

        self.logger.info('Device [%s]' % self.device)

    def train(
            self,
            samples,
            max_iters=5000,
            log_interval=50,
            save_interval=50,
            noise=0.0,
            validation_fraction=0.1):

        start_time = time.time()

        if self.path:
            fig, ax = plt.subplots()
            ax.scatter(samples[:, 0], samples[:, 1])
            self.writer.add_figure('originals', fig, self.total_iters)
            np.save(
                os.path.join(self.path, 'data', 'originals.npy'),
                samples)

        if noise < 0:
            kdt = scipy.spatial.cKDTree(samples)
            dists, neighs = kdt.query(samples, 2)
            training_noise = .2 * np.mean(dists)
        else:
            training_noise = noise

        if self.log:
            self.logger.info('Number of training samples [%d]' % samples.shape[0])
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
        best_model = copy.deepcopy(self.netG)

        for epoch in range(1, max_iters + 1):

            self.total_iters += 1

            train_loss = self._train(
                epoch, self.netG, train_loader, noise=training_noise)
            validation_loss = self._validate(epoch, self.netG, valid_loader)

            if validation_loss < best_validation_loss:
                best_validation_epoch = epoch
                best_validation_loss = validation_loss
                best_model = copy.deepcopy(self.netG)

            if epoch == 1 or epoch % log_interval == 0:
                self.logger.info('Epoch [%i] train loss [%5.4f] validation loss [%5.4f]' % (
                    epoch, train_loss, validation_loss))

            if self.path:
                self.writer.add_scalar('loss', validation_loss, self.total_iters)
                if epoch % save_interval == 0:
                    torch.save(
                        self.netG.state_dict(),
                        os.path.join(self.path, 'models', 'netG.pt')
                    )
                    self._train_plot(self.netG, samples)

        self.logger.info('Best epoch [%i] validation loss [%5.4f]' % (best_validation_epoch, best_validation_loss))

        self.netG.load_state_dict(best_model.state_dict())

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

        # Transpose so shape is (chain_num, iteration, dim)
        samples = np.transpose(np.array(samples), axes=[1, 0, 2])
        likes = np.transpose(np.array(likes), axes=[1, 0])
        latent = np.transpose(np.array(latent), axes=[1, 0, 2])

        if self.path and plot:
            cmap = plt.cm.jet
            cmap.set_under('w', 1)
            fig, ax = plt.subplots()
            ax.hist2d(samples[:, -1, 0], samples[:, -1, 1],
                      bins=200, cmap=cmap, vmin=1, alpha=0.2)
            if self.writer is not None:
                self.writer.add_figure('chain', fig, self.total_iters)

        if out_chain is not None:
            for ib in range(batch_size):
                files[ib].close()

        return samples, likes, latent, scale, ncall

    def _jacobian(self, z):
        """ Calculate det d f^{-1} (z)/dz
        """
        z.requires_grad_(True)
        x, _ = self.netG(z, mode='inverse')
        J = torch.stack([torch.autograd.grad(outputs=x[:, i], inputs=z, retain_graph=True,
                                             grad_outputs=torch.ones(z.shape[0]))[0] for i in
                         range(x.shape[1])]).permute(1, 0, 2)
        return torch.stack([torch.log(torch.abs(torch.det(J[i, :, :])))
                            for i in range(x.shape[0])])

    def _train(self, epoch, model, loader, noise=0.0):

        model.train()
        train_loss = 0

        for batch_idx, data in enumerate(loader):
            if isinstance(data, list):
                if len(data) > 1:
                    cond_data = data[1].float()
                    cond_data = cond_data.to(self.device)
                else:
                    cond_data = None
                data = data[0]
            data = (data + noise * torch.randn_like(data)).to(self.device)
            self.optimizer.zero_grad()
            loss = -model.log_probs(data, cond_data).mean()
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        for module in model.modules():
            if isinstance(module, BatchNormFlow):
                module.momentum = 0

        with torch.no_grad():
            model(loader.dataset.tensors[0].to(data.device))

        for module in model.modules():
            if isinstance(module, BatchNormFlow):
                module.momentum = 1

        return train_loss / len(loader.dataset)

    def _validate(self, epoch, model, loader):

        model.eval()
        val_loss = 0
        cond_data = None

        for batch_idx, data in enumerate(loader):
            if isinstance(data, list):
                if len(data) > 1:
                    cond_data = data[1].float()
                    cond_data = cond_data.to(self.device)
                data = data[0]
            data = data.to(self.device)
            with torch.no_grad():
                # sum up batch loss
                val_loss += -model.log_probs(data, cond_data).mean().item()

        return val_loss / len(loader.dataset)

    def _train_plot(self, model, dataset, plot_grid=True):

        model.eval()
        with torch.no_grad():
            x_synth = model.sample(dataset.size).detach().cpu().numpy()
            z, _ = model(torch.from_numpy(dataset).float().to(self.device))
            z = z.detach().cpu().numpy()
            if plot_grid and self.x_dim == 2:
                grid = []
                for x in np.linspace(np.min(dataset[:, 0]), np.max(dataset[:, 0]), 10):
                    for y in np.linspace(np.min(dataset[:, 1]), np.max(dataset[:, 1]), 5000):
                        grid.append([x, y])
                for y in np.linspace(np.min(dataset[:, 1]), np.max(dataset[:, 1]), 10):
                    for x in np.linspace(np.min(dataset[:, 0]), np.max(dataset[:, 0]), 5000):
                        grid.append([x, y])
                grid = np.array(grid)
                z_grid, _ = model(torch.from_numpy(grid).float().to(self.device))
                z_grid = z_grid.detach().cpu().numpy()

        if self.writer is not None:
            fig, ax = plt.subplots(2, figsize=(5, 10))
            ax[0].scatter(dataset[:, 0], dataset[:, 1], c=dataset[:, 0], s=4)
            ax[1].scatter(z[:, 0], z[:, 1], c=dataset[:, 0], s=4)
            self.writer.add_figure('latent', fig, self.total_iters)

        if self.path:
            fig, ax = plt.subplots(1, 3, figsize=(10, 4))
            if plot_grid and self.x_dim == 2:
                ax[0].scatter(grid[:, 0], grid[:, 1], c=grid[:, 0], marker='.', s=1, linewidths=0)
            ax[0].scatter(dataset[:, 0], dataset[:, 1], s=4)
            ax[0].set_title('Real data')
            if plot_grid and self.x_dim == 2:
                ax[1].scatter(z_grid[:, 0], z_grid[:, 1], c=grid[:, 0], marker='.', s=1, linewidths=0)
            ax[1].scatter(z[:, 0], z[:, 1], s=4)
            ax[1].set_title('Latent data')
            ax[1].set_xlim([-3, 3])
            ax[1].set_ylim([-3, 3])
            ax[2].scatter(x_synth[:, 0], x_synth[:, 1], s=2)
            ax[2].set_title('Synthetic data')
            plt.tight_layout()
            plt.savefig(os.path.join(self.path, 'plots', 'plot_%s.png' % self.total_iters))
            plt.close()

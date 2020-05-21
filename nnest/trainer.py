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
import logging

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.spatial
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nnest.networks import SingleSpeedNVP, FastSlowNVP, SingleSpeedSpline, FastSlowSpline
from nnest.utils.logger import create_logger


class Trainer(object):

    def __init__(self,
                 x_dim,
                 hidden_dim,
                 num_slow=0,
                 batch_size=100,
                 flow='nvp',
                 scale='',
                 num_blocks=5,
                 num_layers=2,
                 base_dist=None,
                 train=True,
                 load_model='',
                 log_dir='logs',
                 use_gpu=False,
                 log=True
                 ):

        self.device = torch.device(
            'cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

        self.x_dim = x_dim
        self.z_dim = x_dim

        self.batch_size = batch_size
        self.total_iters = 0

        assert x_dim > num_slow
        num_fast = x_dim - num_slow
        self.num_slow = num_slow

        if flow.lower() == 'nvp':
            if num_slow > 0:
                self.netG = FastSlowNVP(num_fast, num_slow, hidden_dim, num_blocks, num_layers, device=self.device,
                                        scale=scale, prior=base_dist)
            else:
                self.netG = SingleSpeedNVP(x_dim, hidden_dim, num_blocks, num_layers, device=self.device,
                                           scale=scale, prior=base_dist)
        elif flow.lower() == 'spline':
            if num_slow > 0:
                self.netG = FastSlowSpline(num_fast, num_slow, hidden_dim, num_blocks, device=self.device,
                                           prior=base_dist)
            else:
                self.netG = SingleSpeedSpline(x_dim, hidden_dim, num_blocks,
                                              device=self.device, prior=base_dist)
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

        self.logger = create_logger(__name__, level=logging.INFO)
        self.log = log

        if self.path is not None:
            self.logger.info(self.netG)
            self.writer = SummaryWriter(self.path)

        self.logger.info('Number of network params: [%s]' % sum(p.numel() for p in self.netG.parameters()))
        self.logger.info('Device [%s]' % self.device)

    def train(
            self,
            samples,
            max_iters=5000,
            log_interval=50,
            save_interval=50,
            jitter=0.0,
            validation_fraction=0.1,
            patience=50):

        start_time = time.time()

        if self.path:
            fig, ax = plt.subplots()
            ax.scatter(samples[:, 0], samples[:, 1])
            self.writer.add_figure('originals', fig, self.total_iters)
            np.save(
                os.path.join(self.path, 'data', 'originals.npy'),
                samples)

        if jitter < 0:
            kdt = scipy.spatial.cKDTree(samples)
            dists, neighs = kdt.query(samples, 2)
            training_jitter = .2 * np.mean(dists)
        else:
            training_jitter = jitter

        if self.log:
            self.logger.info('Number of training samples [%d]' % samples.shape[0])
            self.logger.info('Training jitter [%5.4f]' % training_jitter)

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

        counter = 0

        for epoch in range(1, max_iters + 1):

            self.total_iters += 1

            train_loss = self._train(epoch, train_loader, jitter=training_jitter)
            validation_loss = self._validate(epoch, valid_loader)

            if validation_loss < best_validation_loss:
                best_validation_epoch = epoch
                best_validation_loss = validation_loss
                best_model = copy.deepcopy(self.netG)
                counter = 0

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
                    self._train_plot(samples)

            counter += 1

            if counter > patience:
                self.logger.info('Epoch [%i] ran out of patience' % (epoch))
                break

        self.logger.info('Best epoch [%i] validation loss [%5.4f]' % (best_validation_epoch, best_validation_loss))

        self.netG.load_state_dict(best_model.state_dict())

    def get_latent_samples(self, x):
        if type(x) is np.ndarray:
            z, _ = self.netG.forward(torch.from_numpy(x).float().to(self.device))
        else:
            z, _ = self.netG.forward(x)
        z = z.detach().cpu().numpy()
        return z

    def get_samples(self, z):
        if type(z) is np.ndarray:
            x, _ = self.netG.inverse(torch.from_numpy(z).float().to(self.device))
        else:
            x, _ = self.netG.inverse(z)
        x = x.detach().cpu().numpy()
        return x

    def _jacobian(self, z):
        """ Calculate det d f^{-1} (z)/dz
        """
        z.requires_grad_(True)
        x, _ = self.netG.inverse(z)
        J = torch.stack([torch.autograd.grad(outputs=x[:, i], inputs=z, retain_graph=True,
                                             grad_outputs=torch.ones(z.shape[0]))[0] for i in
                         range(x.shape[1])]).permute(1, 0, 2)
        return torch.stack([torch.log(torch.abs(torch.det(J[i, :, :])))
                            for i in range(x.shape[0])])

    def _train(self, epoch, loader, jitter=0.0):

        self.netG.train()
        train_loss = 0

        for batch_idx, data in enumerate(loader):
            if isinstance(data, list):
                data = data[0]
            data = (data + jitter * torch.randn_like(data)).to(self.device)
            self.optimizer.zero_grad()
            loss = -self.netG.log_probs(data).mean()
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            self.netG(loader.dataset.tensors[0].to(data.device))

        return train_loss / len(loader.dataset)

    def _validate(self, epoch, loader):

        self.netG.eval()
        val_loss = 0

        for batch_idx, data in enumerate(loader):
            if isinstance(data, list):
                data = data[0]
            data = data.to(self.device)
            with torch.no_grad():
                # sum up batch loss
                val_loss += -self.netG.log_probs(data).mean().item()

        return val_loss / len(loader.dataset)

    def _train_plot(self, samples, plot_grid=True):

        self.netG.eval()
        with torch.no_grad():
            x_synth = self.netG.sample(samples.size).detach().cpu().numpy()
            z = self.get_latent_samples(samples)
            if plot_grid and self.x_dim == 2:
                grid = []
                for x in np.linspace(np.min(samples[:, 0]), np.max(samples[:, 0]), 10):
                    for y in np.linspace(np.min(samples[:, 1]), np.max(samples[:, 1]), 5000):
                        grid.append([x, y])
                for y in np.linspace(np.min(samples[:, 1]), np.max(samples[:, 1]), 10):
                    for x in np.linspace(np.min(samples[:, 0]), np.max(samples[:, 0]), 5000):
                        grid.append([x, y])
                grid = np.array(grid)
                z_grid = self.get_latent_samples(grid)

        if self.writer is not None:
            fig, ax = plt.subplots(2, figsize=(5, 10))
            ax[0].scatter(samples[:, 0], samples[:, 1], c=samples[:, 0], s=4)
            ax[1].scatter(z[:, 0], z[:, 1], c=samples[:, 0], s=4)
            self.writer.add_figure('latent', fig, self.total_iters)

        if self.path:
            fig, ax = plt.subplots(1, 3, figsize=(10, 4))
            if plot_grid and self.x_dim == 2:
                ax[0].scatter(grid[:, 0], grid[:, 1], c=grid[:, 0], marker='.', s=1, linewidths=0)
            ax[0].scatter(samples[:, 0], samples[:, 1], s=4)
            ax[0].set_title('Real data')
            if plot_grid and self.x_dim == 2:
                ax[1].scatter(z_grid[:, 0], z_grid[:, 1], c=grid[:, 0], marker='.', s=1, linewidths=0)
            ax[1].scatter(z[:, 0], z[:, 1], s=4)
            ax[1].set_title('Latent data')
            ax[1].set_xlim([-np.max(np.abs(z)), np.max(np.abs(z))])
            ax[1].set_ylim([-np.max(np.abs(z)), np.max(np.abs(z))])
            ax[2].scatter(x_synth[:, 0], x_synth[:, 1], s=2)
            ax[2].set_title('Synthetic data')
            plt.tight_layout()
            plt.savefig(os.path.join(self.path, 'plots', 'plot_%s.png' % self.total_iters))
            plt.close()

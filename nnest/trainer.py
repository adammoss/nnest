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
import matplotlib.pyplot as plt
from matplotlib import collections as mc

from nnest.networks import SingleSpeedCholeksy, SingleSpeedNVP, FastSlowNVP, SingleSpeedSpline, FastSlowSpline
from nnest.utils.logger import create_logger


class Trainer(object):
    best_validation_epoch = None
    best_validation_loss = None

    def __init__(self,
                 x_dim,
                 hidden_dim=16,
                 num_slow=0,
                 batch_size=100,
                 flow='spline',
                 scale='',
                 num_blocks=3,
                 num_layers=1,
                 base_dist=None,
                 load_model='',
                 log_dir='logs/test',
                 use_gpu=False,
                 log=True,
                 learning_rate=0.0001,
                 weight_decay=1e-6,
                 log_level=logging.INFO):
        """

        Args:
            x_dim:
            hidden_dim:
            num_slow:
            batch_size:
            flow:
            scale:
            num_blocks:
            num_layers:
            base_dist:
            load_model:
            log_dir:
            use_gpu:
            log:
            learning_rate:
            weight_decay:
            log_level:
        """

        self.device = torch.device(
            'cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

        self.x_dim = x_dim
        self.z_dim = x_dim

        self.batch_size = batch_size
        self.total_iters = 0

        assert x_dim > num_slow
        num_fast = x_dim - num_slow
        self.num_slow = num_slow

        if flow.lower() == 'choleksy':
            self.netG = SingleSpeedCholeksy(x_dim, device=self.device, prior=base_dist)
        elif flow.lower() == 'nvp':
            if num_slow > 0:
                self.netG = FastSlowNVP(num_fast, num_slow, hidden_dim, num_blocks, num_layers, scale=scale,
                                        device=self.device, prior=base_dist)
            else:
                self.netG = SingleSpeedNVP(x_dim, hidden_dim, num_blocks, num_layers, scale=scale,
                                           device=self.device, prior=base_dist)
        elif flow.lower() == 'spline':
            if num_slow > 0:
                self.netG = FastSlowSpline(num_fast, num_slow, hidden_dim, num_blocks, tail_bound=3,
                                           device=self.device, prior=base_dist)
            else:
                self.netG = SingleSpeedSpline(x_dim, hidden_dim, num_blocks, tail_bound=3,
                                              device=self.device, prior=base_dist)
        else:
            raise NotImplementedError

        if load_model:
            self.path = os.path.join(log_dir, load_model)
            self.netG.load_state_dict(torch.load(
                os.path.join(self.path, 'models', 'netG.pt')
            ))
        else:
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

        self.optimizer = torch.optim.Adam(
            self.netG.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.logger = create_logger(__name__, level=log_level)
        self.log = log

        if self.path is not None:
            self.logger.info(self.netG)
            self.writer = SummaryWriter(self.path)

        self.logger.info('Number of network params: [%s]' % sum(p.numel() for p in self.netG.parameters()))
        self.logger.info('Device [%s]' % self.device)

    def train(
            self,
            samples,
            max_iters=10000,
            log_interval=100,
            save_interval=100,
            jitter=0.0,
            validation_fraction=0.1,
            patience=50,
            l2_norm=0.0):
        """

        Args:
            samples:
            max_iters:
            log_interval:
            save_interval:
            plot_interval:
            jitter:
            validation_fraction:
            patience:
            l2_norm:
        """

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

            train_loss = self._train(epoch, train_loader, jitter=training_jitter, l2_norm=l2_norm)
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

            counter += 1

            if counter > patience:
                self.logger.info('Epoch [%i] ran out of patience' % (epoch))
                if self.path:
                    torch.save(
                        self.netG.state_dict(),
                        os.path.join(self.path, 'models', 'netG.pt')
                    )
                break

        end_time = time.time()

        self.logger.info('Best epoch [%i] validation loss [%5.4f] train time (s) [%5.4f]]'
                         % (best_validation_epoch, best_validation_loss, end_time - start_time))
        self.best_validation_epoch = best_validation_epoch
        self.best_validation_loss = best_validation_loss

        self.netG.load_state_dict(best_model.state_dict())

        if self.path:
            self.plot_samples(
                samples, outfile=os.path.join(self.path, 'plots', 'plot_%s.png' % self.total_iters))

    def forward(self, x, to_numpy=False):
        if type(x) is np.ndarray:
            z, log_det_J = self.netG.forward(torch.from_numpy(x).float().to(self.device))
        else:
            z, log_det_J = self.netG.forward(x)
        z = z.detach()
        log_det_J = log_det_J.detach()
        if to_numpy:
            z = z.cpu().numpy()
            log_det_J = log_det_J.cpu().numpy()
        return z, log_det_J

    def inverse(self, z, to_numpy=False):
        if type(z) is np.ndarray:
            x, log_det_J = self.netG.inverse(torch.from_numpy(z).float().to(self.device))
        else:
            x, log_det_J = self.netG.inverse(z)
        x = x.detach()
        log_det_J = log_det_J.detach()
        if to_numpy:
            x = x.cpu().numpy()
            log_det_J = log_det_J.cpu().numpy()
        return x, log_det_J

    def get_prior_samples(self, num_samples, to_numpy=False):
        z = self.netG.prior.sample((num_samples,)).to(self.device)
        z = z.detach()
        if to_numpy:
            z = z.cpu().numpy()
        return z

    def get_latent_samples(self, x, to_numpy=False):
        z, _ = self.forward(x, to_numpy=to_numpy)
        return z

    def get_samples(self, z, to_numpy=False):
        x, _ = self.inverse(z, to_numpy=to_numpy)
        return x

    def get_synthetic_samples(self, num_samples, to_numpy=False):
        x = self.netG.sample(num_samples)
        x = x.detach()
        if to_numpy:
            x = x.cpu().numpy()
        return x

    def log_probs(self, x, to_numpy=False):
        if type(x) is np.ndarray:
            log_probs = self.netG.log_probs(torch.from_numpy(x).float().to(self.device))
        else:
            log_probs = self.netG.log_probs(x)
        log_probs = log_probs.detach()
        if to_numpy:
            log_probs = log_probs.cpu().numpy()
        return log_probs

    def plot_samples(self, samples, outfile=None, plot_synthetic=True):

        self.netG.eval()

        with torch.no_grad():
            if plot_synthetic:
                fig, ax = plt.subplots(1, 3, figsize=(12, 5))
            else:
                fig, ax = plt.subplots(1, 2, figsize=(9, 5))
            ax[0].scatter(samples[:, 0], samples[:, 1], c='r', s=5, alpha=0.5)
            ax[0].set_title('Real data')
            ax[0].set_xlim([np.min(samples[:, 0]) - 0.1, np.max(samples[:, 0]) + 0.1])
            ax[0].set_ylim([np.min(samples[:, 1]) - 0.1, np.max(samples[:, 1]) + 0.1])
            if self.x_dim == 2:
                ng = 30
                xx = np.linspace(-3, 3, ng)
                yy = np.linspace(-3, 3, ng)
                xv, yv = np.meshgrid(xx, yy)
                xy = np.stack([xv, yv], axis=-1)
                xy = xy.reshape((ng * ng, 2))
                xy = torch.from_numpy(xy.astype(np.float32))
                xs = self.get_samples(xy, to_numpy=True)
                xs = xs.reshape((ng, ng, 2))
                p1 = np.reshape(xs[1:, :, :], (ng ** 2 - ng, 2))
                p2 = np.reshape(xs[:-1, :, :], (ng ** 2 - ng, 2))
                lcy = mc.LineCollection(zip(p1, p2), linewidths=1, alpha=0.2, color='k')
                p1 = np.reshape(xs[:, 1:, :], (ng ** 2 - ng, 2))
                p2 = np.reshape(xs[:, :-1, :], (ng ** 2 - ng, 2))
                lcx = mc.LineCollection(zip(p1, p2), linewidths=1, alpha=0.2, color='k')
                ax[0].add_collection(lcy)
                ax[0].add_collection(lcx)
            z = self.get_latent_samples(samples, to_numpy=True)
            ax[1].scatter(z[:, 0], z[:, 1], c='r', s=5, alpha=0.5)
            ax[1].set_title('Latent data')
            ax[1].set_xlim([np.min(z[:, 0]) - 0.1, np.max(z[:, 0]) + 0.1])
            ax[1].set_ylim([np.min(z[:, 1]) - 0.1, np.max(z[:, 1]) + 0.1])
            if self.x_dim == 2:
                ng = 30
                xx = np.linspace(np.min(samples[:, 0]) - 0.1, np.max(samples[:, 0] + 0.1), ng)
                yy = np.linspace(np.min(samples[:, 1]) - 0.1, np.max(samples[:, 1] + 0.1), ng)
                xv, yv = np.meshgrid(xx, yy)
                xy = np.stack([xv, yv], axis=-1)
                xy = xy.reshape((ng * ng, 2))
                xy = torch.from_numpy(xy.astype(np.float32))
                xs = self.get_latent_samples(xy, to_numpy=True)
                xs = xs.reshape((ng, ng, 2))
                p1 = np.reshape(xs[1:, :, :], (ng ** 2 - ng, 2))
                p2 = np.reshape(xs[:-1, :, :], (ng ** 2 - ng, 2))
                lcy = mc.LineCollection(zip(p1, p2), linewidths=1, alpha=0.2, color='k')
                p1 = np.reshape(xs[:, 1:, :], (ng ** 2 - ng, 2))
                p2 = np.reshape(xs[:, :-1, :], (ng ** 2 - ng, 2))
                lcx = mc.LineCollection(zip(p1, p2), linewidths=1, alpha=0.2, color='k')
                ax[1].add_collection(lcy)
                ax[1].add_collection(lcx)
            if plot_synthetic:
                x = self.get_synthetic_samples(samples.size, to_numpy=True)
                ax[2].scatter(x[:, 0], x[:, 1], c='r', s=5, alpha=0.5)
                ax[2].set_title('Synthetic data')
            plt.tight_layout()
            if outfile is not None:
                plt.savefig(outfile)
                plt.close()
            else:
                plt.show()
            if self.writer is not None:
                fig, ax = plt.subplots(2, figsize=(5, 10))
                ax[0].scatter(samples[:, 0], samples[:, 1], c='r', s=5, alpha=0.5)
                ax[1].scatter(z[:, 0], z[:, 1], c='r', s=5, alpha=0.5)
                self.writer.add_figure('latent', fig, self.total_iters)

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

    def _train(self, epoch, loader, jitter=0.0, l2_norm=0.0):

        self.netG.train()
        train_loss = 0

        for batch_idx, data in enumerate(loader):
            if isinstance(data, list):
                data = data[0]
            data = (data + jitter * torch.randn_like(data)).to(self.device)
            self.optimizer.zero_grad()
            loss = -self.netG.log_probs(data).mean()
            l2_loss = 0
            for param in self.netG.parameters():
                l2_loss += (param ** 2).sum()
            train_loss += loss.item()
            loss += l2_norm * l2_loss
            loss.backward()
            self.optimizer.step()

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

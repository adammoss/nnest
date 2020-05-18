import os
import sys
import argparse

import numpy as np
import torch

path = os.path.realpath(os.path.join(os.getcwd(), '../..'))
sys.path.insert(0, path)


def log_gaussian_pdf(theta, sigma=1, mu=0, ndim=None):
    if ndim is None:
        try:
            ndim = len(theta)
        except TypeError:
            assert isinstance(theta, (float, int)), theta
            ndim = 1
    logl = -(np.sum((theta - mu) ** 2) / (2 * sigma ** 2))
    logl -= np.log(2 * np.pi * (sigma ** 2)) * ndim / 2.0
    return logl


class Gaussian(object):

    def __init__(self, sigma=1.0, nderived=0):
        self.sigma = sigma
        self.nderived = nderived

    def __call__(self, theta):
        logl = log_gaussian_pdf(theta, sigma=self.sigma, mu=0)
        return logl, [0.0] * self.nderived


class GaussianMix(object):

    def __init__(self, sep=4, weights=(0.4, 0.3, 0.2, 0.1), sigma=1,
                 nderived=0):
        assert len(weights) in [2, 3, 4], (
            'Weights must have 2, 3 or 4 components. Weights=' + str(weights))
        assert np.isclose(sum(weights), 1), (
            'Weights must sum to 1! Weights=' + str(weights))
        self.nderived = nderived
        self.weights = weights
        self.sigmas = [sigma] * len(weights)
        positions = []
        positions.append(np.asarray([0, sep]))
        positions.append(np.asarray([0, -sep]))
        positions.append(np.asarray([sep, 0]))
        positions.append(np.asarray([-sep, 0]))
        self.positions = positions[:len(weights)]

    def __call__(self, theta):
        thetas = []
        for pos in self.positions:
            thetas.append(copy.deepcopy(theta))
            thetas[-1][:2] -= pos
        logls = [(Gaussian(sigma=self.sigmas[i])(thetas[i])[0]
                  + np.log(self.weights[i])) for i in range(len(self.weights))]
        logl = scipy.special.logsumexp(logls)
        return logl, [0.0] * self.nderived



def main(args):

    from nnest.trainer import Trainer
    from nnest.distributions import GeneralisedNormal

    g = GaussianMix()

    def loglike(z):
        return np.array([g(x)[0] for x in z])

    def transform(x):
        return 10. * x

    n_samples = args.num_live_points
    fraction = args.fraction

    x = 2 * (np.random.uniform(size=(int(n_samples / fraction), 2)) - 0.5)
    likes = loglike(transform(x))
    idx = np.argsort(-likes)
    samples = x[idx[0:n_samples]]

    if args.base_dist == 'gen_normal':
        base_dist = GeneralisedNormal(torch.zeros(args.x_dim), torch.ones(args.x_dim), torch.tensor(args.beta))
    else:
        base_dist = None

    t = Trainer(args.x_dim, args.hidden_dim, log_dir=args.log_dir,  num_blocks=args.num_blocks,
                num_layers=args.num_layers, base_dist=base_dist, scale=args.scale)
    t.train(samples, max_iters=args.train_iters)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--x_dim', type=int, default=2,
                        help="Dimensionality")
    parser.add_argument('--train_iters', type=int, default=1000,
                        help="number of train iters")
    parser.add_argument("--num_live_points", type=int, default=1000)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('-use_gpu', action='store_true')
    parser.add_argument('--flow', type=str, default='nvp')
    parser.add_argument('--num_blocks', type=int, default=5)
    parser.add_argument('--num_slow', type=int, default=0)
    parser.add_argument('--corr', type=float, default=0.99)
    parser.add_argument('--log_dir', type=str, default='logs/flow/gauss')
    parser.add_argument('--beta', type=float, default=8.0)
    parser.add_argument('--base_dist', type=str, default='')
    parser.add_argument('--scale', type=str, default='')
    parser.add_argument('--fraction', type=float, default=0.02)

    args = parser.parse_args()
    main(args)

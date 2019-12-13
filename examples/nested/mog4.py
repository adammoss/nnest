import os
import sys
import argparse
import copy

import numpy as np
import scipy.special
import torch

sys.path.append(os.getcwd())


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

    def __init__(self, sep=4, weights=(0.4, 0.3, 0.2, 0.1), sigma=1, nderived=0):
        assert len(weights) in [2, 3, 4], ('Weights must have 2, 3 or 4 components. Weights=' + str(weights))
        assert np.isclose(sum(weights), 1), ('Weights must sum to 1! Weights=' + str(weights))
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

    from nnest import NestedSampler
    from nnest.distributions import GeneralisedNormal

    g = GaussianMix()

    def loglike(z):
        return np.array([g(x)[0] for x in z])

    def transform(x):
        return 10. * x

    if args.base_dist == 'gen_normal':
        base_dist = GeneralisedNormal(torch.zeros(args.x_dim), torch.ones(args.x_dim), torch.tensor(args.beta))
    else:
        base_dist = None

    sampler = NestedSampler(args.x_dim, loglike, transform=transform, log_dir=args.log_dir, num_live_points=args.num_live_points,
                            hidden_dim=args.hidden_dim, num_layers=args.num_layers, num_blocks=args.num_blocks, num_slow=args.num_slow,
                            use_gpu=args.use_gpu, base_dist=base_dist, scale=args.scale)
    sampler.run(train_iters=args.train_iters, mcmc_steps=args.mcmc_steps, volume_switch=args.switch, noise=args.noise)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--x_dim', type=int, default=5,
                        help="Dimensionality")
    parser.add_argument('--train_iters', type=int, default=2000,
                        help="number of train iters")
    parser.add_argument("--mcmc_steps", type=int, default=0)
    parser.add_argument("--num_live_points", type=int, default=1000)
    parser.add_argument('--switch', type=float, default=-1)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('-use_gpu', action='store_true')
    parser.add_argument('--flow', type=str, default='nvp')
    parser.add_argument('--num_blocks', type=int, default=5)
    parser.add_argument('--noise', type=float, default=-1)
    parser.add_argument("--test_samples", type=int, default=0)
    parser.add_argument('--num_slow', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='logs/mog4')
    parser.add_argument('--base_dist', type=str, default='')
    parser.add_argument('--scale', type=str, default='constant')
    parser.add_argument('--beta', type=float, default=8.0)

    args = parser.parse_args()
    main(args)

    print('Expected log Z: %5.4f' % (args.x_dim * np.log(20)))

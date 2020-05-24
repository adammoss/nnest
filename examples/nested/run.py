import os
import sys
import argparse

import numpy as np
import torch

path = os.path.realpath(os.path.join(os.getcwd(), '../..'))
sys.path.insert(0, path)


def main(args):

    from nnest import NestedSampler
    from nnest.distributions import GeneralisedNormal
    from nnest.likelihoods import Himmelblau, Rosenbrock, Gaussian, Eggbox, GaussianShell, GaussianMix

    if args.base_dist == 'gen_normal':
        base_dist = GeneralisedNormal(torch.zeros(args.x_dim), torch.ones(args.x_dim), torch.tensor(args.beta))
    else:
        base_dist = None

    if args.likelihood.lower() == 'himmelblau':
        like = Himmelblau()
        transform = lambda x: 5 * x
    elif args.likelihood.lower() == 'rosenbrock':
        like = Rosenbrock(args.x_dim)
        transform = lambda x: 5*x
    elif args.likelihood.lower() == 'gaussian':
        like = Gaussian(args.x_dim, args.corr)
        transform = lambda x: 3 * x
    elif args.likelihood.lower() == 'eggbox':
        like = Eggbox()
        transform = lambda x: x * 5 * np.pi
    elif args.likelihood.lower() == 'shell':
        like = GaussianShell(args.x_dim)
        transform = lambda x: 5 * x
    elif args.likelihood.lower() == 'mixture':
        like = GaussianMix(args.x_dim)
        transform = lambda x: 10 * x
    else:
        raise ValueError('Likelihood not found')

    log_dir = os.path.join(args.log_dir, args.likelihood)
    log_dir += args.log_suffix

    sampler = NestedSampler(like.x_dim, like.loglike, transform=transform, log_dir=log_dir,
                            num_live_points=args.num_live_points, hidden_dim=args.hidden_dim,
                            num_layers=args.num_layers, num_blocks=args.num_blocks, num_slow=args.num_slow,
                            use_gpu=args.use_gpu, base_dist=base_dist, scale=args.scale, flow=args.flow)
    sampler.run(train_iters=args.train_iters, mcmc_steps=args.mcmc_steps, volume_switch=args.switch, jitter=args.jitter,
                num_test_mcmc_samples=args.test_samples, test_mcmc_steps=args.test_mcmc_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--x_dim', type=int, default=2,
                        help="Dimensionality")
    parser.add_argument('--train_iters', type=int, default=2000,
                        help="number of train iters")
    parser.add_argument("--mcmc_steps", type=int, default=0)
    parser.add_argument("--num_live_points", type=int, default=1000)
    parser.add_argument('--switch', type=float, default=-1)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('-use_gpu', action='store_true')
    parser.add_argument('--flow', type=str, default='spline')
    parser.add_argument('--num_blocks', type=int, default=3)
    parser.add_argument('--jitter', type=float, default=-1)
    parser.add_argument("--test_samples", type=int, default=0)
    parser.add_argument("--test_mcmc_steps", type=int, default=1000)
    parser.add_argument('--num_slow', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--likelihood', type=str, default='rosenbrock')
    parser.add_argument('--log_suffix', type=str, default='')
    parser.add_argument('--base_dist', type=str, default='')
    parser.add_argument('--scale', type=str, default='')
    parser.add_argument('--beta', type=float, default=8.0)
    parser.add_argument('--corr', type=float, default=0.99)

    args = parser.parse_args()
    main(args)

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

    def loglike(z):
        return np.array([-sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0) for x in z])

    def transform(x):
        return 5. * x

    if args.base_dist == 'gen_normal':
        base_dist = GeneralisedNormal(torch.zeros(args.x_dim), torch.ones(args.x_dim), torch.tensor(args.beta))
    else:
        base_dist = None

    sampler = NestedSampler(args.x_dim, loglike, transform=transform, log_dir=args.log_dir, num_live_points=args.num_live_points,
                            hidden_dim=args.hidden_dim, num_layers=args.num_layers, num_blocks=args.num_blocks, num_slow=args.num_slow,
                            use_gpu=args.use_gpu, base_dist=base_dist, scale=args.scale)
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
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('-use_gpu', action='store_true')
    parser.add_argument('--flow', type=str, default='nvp')
    parser.add_argument('--num_blocks', type=int, default=5)
    parser.add_argument('--jitter', type=float, default=-1)
    parser.add_argument("--test_samples", type=int, default=0)
    parser.add_argument("--test_mcmc_steps", type=int, default=1000)
    parser.add_argument('--num_slow', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='logs/rosenbrock')
    parser.add_argument('--base_dist', type=str, default='')
    parser.add_argument('--scale', type=str, default='')
    parser.add_argument('--beta', type=float, default=8.0)

    args = parser.parse_args()
    main(args)

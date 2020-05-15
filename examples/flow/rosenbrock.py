import os
import sys
import argparse

import numpy as np
import torch

sys.path.append(os.getcwd())


def main(args):

    from nnest.trainer import Trainer
    from nnest.distributions import GeneralisedNormal

    def loglike(z):
        return np.array([-sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0) for x in z])

    def transform(x):
        return 5. * x

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
    parser.add_argument('--noise', type=float, default=-1)
    parser.add_argument('--num_slow', type=int, default=0)
    parser.add_argument('--corr', type=float, default=0.99)
    parser.add_argument('--log_dir', type=str, default='logs/flow/rosenbrock')
    parser.add_argument('--beta', type=float, default=8.0)
    parser.add_argument('--base_dist', type=str, default='')
    parser.add_argument('--scale', type=str, default='constant')
    parser.add_argument('--fraction', type=float, default=0.02)

    args = parser.parse_args()
    main(args)

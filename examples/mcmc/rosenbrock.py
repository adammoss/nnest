import os
import sys
import argparse

import numpy as np

sys.path.append(os.getcwd())


def main(args):
    from nnest.mcmc import MCMCSampler

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    def loglike_orig(z):
        return np.array([-sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0) for x in z])
    
    # pairwise rosenbrocks, multiplied together
    assert args.x_dim % 2 == 0
    
    def loglike(z):
        return np.array([-sum(100.0 * (x[1::2] - x[::2] ** 2.0) ** 2.0 + (1 - x[::2]) ** 2.0) * 2. / len(x) for x in z])

    def transform(x):
        return 500. * x - 100

    sampler = MCMCSampler(args.x_dim, loglike, transform=transform, log_dir=args.log_dir, hidden_dim=args.hidden_dim,
                          num_layers=args.num_layers, num_blocks=args.num_blocks, num_slow=args.num_slow)
    sampler.run(train_iters=args.train_iters, mcmc_steps=args.mcmc_steps, single_thin=10, 
                bootstrap_iters=args.burnin_iters, bootstrap_mcmc_steps=5000 + 1000 * args.x_dim)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--x_dim', type=int, default=2,
                        help="Dimensionality")
    parser.add_argument('--train_iters', type=int, default=100,
                        help="number of train iters")
    parser.add_argument('--burnin_iters', type=int, default=1,
                        help="number of iters for finding good training spot")
    parser.add_argument("--mcmc_steps", type=int, default=10000)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('-use_gpu', action='store_true')
    parser.add_argument('--flow', type=str, default='nvp')
    parser.add_argument('--num_blocks', type=int, default=5)
    parser.add_argument('--noise', type=float, default=-1)
    parser.add_argument('--run_num', type=str, default='')
    parser.add_argument('--num_slow', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='logs/rosenbrock_mcmc')

    args = parser.parse_args()
    main(args)

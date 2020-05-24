import os
import sys
import argparse

sys.path.append(os.getcwd())


def main(args):
    from nnest import MCMCSampler
    from nnest.likelihoods import Himmelblau, Rosenbrock, Gaussian, Eggbox, GaussianShell, GaussianMix

    if args.likelihood.lower() == 'himmelblau':
        like = Himmelblau()
    elif args.likelihood.lower() == 'rosenbrock':
        like = Rosenbrock(args.x_dim)
    elif args.likelihood.lower() == 'gaussian':
        like = Gaussian(args.x_dim, args.corr)
    elif args.likelihood.lower() == 'eggbox':
        like = Eggbox()
    elif args.likelihood.lower() == 'shell':
        like = GaussianShell(args.x_dim)
    elif args.likelihood.lower() == 'mixture':
        like = GaussianMix(args.x_dim)
    else:
        raise ValueError('Likelihood not found')

    log_dir = os.path.join(args.log_dir, args.likelihood)
    log_dir += args.log_suffix

    sampler = MCMCSampler(like.x_dim, like.loglike, log_dir=args.log_dir, hidden_dim=args.hidden_dim,
                          num_layers=args.num_layers, num_blocks=args.num_blocks, num_slow=args.num_slow,
                          use_gpu=args.use_gpu, flow=args.flow)
    sampler.run(train_iters=args.train_iters, mcmc_steps=args.mcmc_steps, single_thin=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--x_dim', type=int, default=2,
                        help="Dimensionality")
    parser.add_argument('--train_iters', type=int, default=2000,
                        help="number of train iters")
    parser.add_argument("--mcmc_steps", type=int, default=10000)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('-use_gpu', action='store_true')
    parser.add_argument('--flow', type=str, default='spline')
    parser.add_argument('--num_blocks', type=int, default=3)
    parser.add_argument('--num_slow', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--likelihood', type=str, default='rosenbrock')
    parser.add_argument('--log_suffix', type=str, default='')
    parser.add_argument('--corr', type=float, default=0.99)

    args = parser.parse_args()
    main(args)

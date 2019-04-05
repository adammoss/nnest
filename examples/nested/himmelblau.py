import os
import sys
import argparse

sys.path.append(os.getcwd())


def main(args):

    from src.nested import NestedSampler

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    def loglike(z):
        z1 = z[:, 0]
        z2 = z[:, 1]
        return - (z1**2 + z2 - 11.)**2 - (z1 + z2**2 - 7.)**2

    def transform(x):
        return 5. * x

    sampler = NestedSampler(loglike, args,  name='himmelblau', transform=transform)
    sampler.run(train_iters=args.train_iters, mcmc_steps=args.mcmc_steps, volume_switch=args.switch, noise=args.noise, num_test_samples=args.test_samples, test_mcmc_steps=args.test_mcmc_steps)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--x_dim', type=int, default=2,
                        help="Dimensionality")
    parser.add_argument('--train_iters', type=int, default=50,
                        help="number of train iters")
    parser.add_argument("--mcmc_steps", type=int, default=0)
    parser.add_argument("--npoints", type=int, default=1000)
    parser.add_argument('--switch', type=float, default=-1)
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('-use_gpu', action='store_true')
    parser.add_argument('--flow', type=str, default='nvp')
    parser.add_argument('--num_blocks', type=int, default=5)
    parser.add_argument('--noise', type=float, default=-1)
    parser.add_argument("--test_samples", type=int, default=0)
    parser.add_argument("--test_mcmc_steps", type=int, default=1000)
    parser.add_argument('--run_num', type=str, default='')
    parser.add_argument('--nslow', type=int, default=0)

    args = parser.parse_args()
    main(args)

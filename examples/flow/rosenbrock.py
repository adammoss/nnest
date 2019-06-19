import os
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())

def main():

    from nnest.trainer import Trainer
    from nnest.distributions import GeneralisedNormal

    def loglike(z):
        return np.array([-sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0) for x in z])

    def transform(x):
        return 5. * x

    xdim = 2
    ndim = 100

    n_samples = 1000
    fraction = 0.02

    x = 2 * (np.random.uniform(size=(int(n_samples / fraction), 2)) - 0.5)
    likes = loglike(transform(x))
    idx = np.argsort(-likes)
    samples = x[idx[0:n_samples]]

    base_dist = GeneralisedNormal(torch.zeros(xdim), torch.ones(xdim), 8)

    t = Trainer(xdim, ndim, log_dir='logs/flow/rosenbrock',  num_blocks=5, num_layers=2, scale='constant',
                base_dist=base_dist)
    t.train(samples, max_iters=1000)


if __name__ == '__main__':
    main()

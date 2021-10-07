import torch
from getdist import plots, MCSamples
import getdist
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal

import os
import sys
from pathlib import Path

sys.path.append(os.getcwd() + "/..")

from nnest.nested import NestedSampler
from likelihoods import *
from priors import UniformPrior
from distributions import GeneralisedNormal

repeats = 2
dims = [2, 4, 5]
methods = ['slice']
v_switch = [-1, -1, -1]

pre = Path(os.getcwd())
post = Path("results/final.csv")

for _ in range(repeats):
    for dim in dims:
        for method in methods:
            like = GaussianMix(dim)
            transform = lambda x: 10*x
            base_dist = MultivariateNormal(torch.zeros(like.x_dim), torch.eye(like.x_dim))

            sampler = NestedSampler(like.x_dim, like, transform=transform, num_live_points=1000,
                                    hidden_dim=32, num_blocks=4, flow='nvp', scale='constant',
                                    base_dist=base_dist, learning_rate=0.01, log_dir="logs/test")

            sampler.run(strategy=['rejection_prior', method],
                        volume_switch=-1, dlogz=0.001, max_iters=1300*dim,
                        train_iters=1000,
                        plot_slice_trace=True)


            path_t = pre / sampler.log_dir / post
            fin = pd.read_csv(path_t)

            if isinstance(like, Rosenbrock):
                fout = pre / "logs/dim_scaling/Rosenbrock.csv"
            elif isinstance(like, Himmelblau):
                fout = pre / "logs/dim_scaling/Himmelblau.csv"
            elif isinstance(like, GaussianMix):
                fout = pre / "logs/dim_scaling/GaussianMix.csv"
            fin.to_csv(fout, mode='a', index=False, header=False)






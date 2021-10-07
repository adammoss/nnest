import torch
from getdist import plots, MCSamples
import getdist
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal

import os
import sys
from pathlib import Path

sys.path.append(os.getcwd() + "/../..")

print(sys.path)

from nnest.nested import NestedSampler
from nnest.likelihoods import *
from nnest.priors import UniformPrior
from nnest.distributions import GeneralisedNormal

like = GaussianMix(3)
transform = lambda x: 10 * x
base_dist = MultivariateNormal(torch.zeros(like.x_dim), torch.eye(like.x_dim))

sampler = NestedSampler(like.x_dim, like, transform=transform, num_live_points=1000,
                        hidden_dim=32, num_blocks=4, flow='nvp', scale='constant',
                        base_dist=base_dist, learning_rate=0.01, log_dir="logs/test")

sampler.run(strategy=['rejection_prior', 'slice'],
            volume_switch=-1, dlogz=0.01, max_iters=6000 * 3,
            train_iters=1000,
            plot_slice_trace=True)

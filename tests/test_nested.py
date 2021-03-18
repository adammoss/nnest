import numpy as np

from nnest import NestedSampler
from nnest.distributions import GeneralisedNormal
from nnest.likelihoods import Himmelblau, Rosenbrock, Gaussian, Eggbox, GaussianShell, GaussianMix

max_evidence_error = 0.2


def test_rosenbrock():
    transform = lambda x: 5 * x
    like = Rosenbrock(2)
    sampler = NestedSampler(2, like, transform=transform,
                            num_live_points=1000, hidden_dim=16,
                            num_layers=1, num_blocks=3, num_slow=0,
                            flow='spline')
    sampler.run(mcmc_num_chains=10, mcmc_dynamic_step_size=False)
    diff = sampler.logz + 5.80
    assert np.abs(diff) <= max_evidence_error

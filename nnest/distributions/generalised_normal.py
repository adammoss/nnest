import math
from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import _standard_normal, broadcast_all
from scipy.stats import gennorm
import numpy as np


class GeneralisedNormal(ExponentialFamily):
    r"""
    Creates a generalised normal (also called Gaussian) distribution parameterized by
    :attr:`loc` and :attr:`scale` and :attr:`beta`.
    Example::
        >>> m = GeneralisedNormal(torch.tensor([0.0]), torch.tensor([1.0]), 2)
    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor):
        beta (float or Tensor):
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive, 'beta': constraints.real}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)

    def __init__(self, loc, scale, beta, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        self.beta = beta
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(GeneralisedNormal, self).__init__(batch_shape, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.tensor(gennorm.rvs(self.beta, size=shape), dtype=torch.float32, device=self.loc.device)

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError

    def usample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return 2 * (np.random.uniform(size=shape) - 0.5)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        log_scale = math.log(self.scale) if isinstance(self.scale, Number) else self.scale.log()
        log_beta = math.log(self.beta) if isinstance(self.beta, Number) else self.beta.log()
        log_gamma = math.log(1.0 / self.beta) if isinstance(self.beta, Number) else torch.mvlgamma(1.0 / self.beta, 1)
        return -((torch.abs(value - self.loc) / (self.scale)) ** self.beta) + log_beta - log_scale - math.log(2) - log_gamma

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    @property
    def _natural_params(self):
        raise NotImplementedError

    def _log_normalizer(self, x, y):
        raise NotImplementedError

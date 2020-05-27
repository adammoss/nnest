import numpy as np


class Prior(object):

    def __init__(self, x_dim):
        self.x_dim = x_dim

    def __call__(self, x):
        if isinstance(x, list):
            x = np.array(x)
        if len(x.shape) > 1:
            return np.array([self.loglike(x) for x in x])
        else:
            return self.loglike(x)

    def loglike(self, x):
        raise NotImplementedError

    def sample(self, x):
        raise NotImplementedError


class UniformPrior(Prior):

    def __init__(self, x_dim, minimum, maximum):
        if not hasattr(minimum, '__len__'):
            self.minimum = np.array([minimum] * x_dim)
        else:
            assert len(minimum) == x_dim
            self.minimum = np.array(minimum)
        if not hasattr(maximum, '__len__'):
            self.maximum = np.array([maximum] * x_dim)
        else:
            assert len(maximum) == x_dim
            self.maximum = np.array(maximum)
        super(UniformPrior, self).__init__(x_dim)

    def __call__(self, x):
        if np.any(x < self.minimum) or np.any(x > self.maximum):
            return -np.inf
        else:
            return 0

    def sample(self, num_samples):
        return self.minimum + (self.maximum - self.minimum) * \
               np.random.uniform(size=(num_samples, self.x_dim))

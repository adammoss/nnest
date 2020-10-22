import copy
import numpy as np
from scipy.stats import multivariate_normal
import scipy.special


class Likelihood(object):
    num_derived = 0
    num_evaluations = 0

    def __init__(self, x_dim):
        self.x_dim = x_dim

    def __call__(self, x):
        if isinstance(x, list):
            x = np.array(x)
        if len(x.shape) > 1:
            self.num_evaluations += x.shape[0]
            return np.array([self.loglike(x) for x in x])
        else:
            self.num_evaluations += 1
            return self.loglike(x)

    def loglike(self, x):
        raise NotImplementedError

    def sample(self, prior, num_samples):
        max_loglike = self.max_loglike
        samples = np.empty((0, self.x_dim))
        while samples.shape[0] < num_samples:
            x = prior.sample(num_samples)
            loglike = self(x)
            ratio = np.exp(loglike - max_loglike)
            r = np.random.uniform(low=0, high=1, size=(num_samples,))
            samples = np.vstack((x[np.where(ratio > r)], samples))
        return samples[0:num_samples]

    def uniform_sample(self, prior, num_samples, fraction):
        x = prior.sample(int(num_samples / fraction))
        loglike = self(x)
        idx = np.argsort(-loglike)
        return x[idx[0:num_samples]], loglike[idx[num_samples-1]]

    def max_loglike(self):
        raise NotImplementedError


class Rosenbrock(Likelihood):

    def loglike(self, x):
        return -sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

    @property
    def max_loglike(self):
        return self(np.ones((self.x_dim,)))

    @property
    def sample_range(self):
        return [-2] * self.x_dim, [12] * self.x_dim


class Himmelblau(Likelihood):
    x_dim = 2

    def __init__(self, x_dim):
        assert self.x_dim == x_dim
        super(Himmelblau, self).__init__(x_dim)

    def loglike(self, x):
        return - (x[0] ** 2 + x[1] - 11.) ** 2 - (x[0] + x[1] ** 2 - 7.) ** 2

    @property
    def max_loglike(self):
        return self([3.0, 2.0])


class Gaussian(Likelihood):

    def __init__(self, x_dim, corr, lim=5):
        self.corr = corr
        self.lim = lim
        super(Gaussian, self).__init__(x_dim)

    def loglike(self, x):
        return multivariate_normal.logpdf(x, mean=np.zeros(self.x_dim),
                                          cov=np.eye(self.x_dim) + self.corr * (1 - np.eye(self.x_dim)))

    @property
    def max_loglike(self):
        return self([0.0] * self.x_dim)

    @property
    def sample_range(self):
        return [-self.lim] * self.x_dim, [self.lim] * self.x_dim


class Eggbox(Likelihood):
    x_dim = 2

    def __init__(self, x_dim):
        assert self.x_dim == x_dim
        super(Eggbox, self).__init__(x_dim)

    def loglike(self, x):
        chi = (np.cos(x[0] / 2.)) * (np.cos(x[1] / 2.))
        return (2. + chi) ** 5

    @property
    def max_loglike(self):
        return self([0.0] * self.x_dim)


class GaussianShell(Likelihood):

    def __init__(self, x_dim, sigma=0.1, rshell=2, center=0):
        self.sigma = sigma
        self.rshell = rshell
        if not hasattr(center, '__len__'):
            self.center = np.array([center] * x_dim)
        elif isinstance(center, list):
            self.center = np.array(center)
        else:
            self.center = center
        super(GaussianShell, self).__init__(x_dim)

    def loglike(self, x):
        rad = np.sqrt(np.sum((self.center - x) ** 2))
        return - ((rad - self.rshell) ** 2) / (2 * self.sigma ** 2)

    @property
    def max_loglike(self):
        return self(self.center - np.array([self.rshell] + [0] * (self.x_dim - 1)))


class DoubleGaussianShell(Likelihood):

    def __init__(self, x_dim, sigmas=[0.1, 0.1], rshells=[2, 2], centers=[-4, 4], weights=[1.0, 1.0]):
        self.shell1 = GaussianShell(x_dim, sigma=sigmas[0], rshell=rshells[0], center=centers[0])
        self.shell2 = GaussianShell(x_dim, sigma=sigmas[1], rshell=rshells[1], center=centers[1])
        self.weights = weights
        super(DoubleGaussianShell, self).__init__(x_dim)

    def loglike(self, x):
        return np.logaddexp(np.log(self.weights[0]) + self.shell1.loglike(x),
                            np.log(self.weights[1]) + self.shell2.loglike(x))

    @property
    def max_loglike(self):
        # This is the worst case scenerio of overlapping shells
        return self.shell1.max_loglike + self.shell2.max_loglike


def log_gaussian_pdf(theta, sigma=1, mu=0, ndim=None):
    if ndim is None:
        try:
            ndim = len(theta)
        except TypeError:
            assert isinstance(theta, (float, int)), theta
            ndim = 1
    logl = -(np.sum((theta - mu) ** 2) / (2 * sigma ** 2))
    logl -= np.log(2 * np.pi * (sigma ** 2)) * ndim / 2.0
    return logl


class GaussianMix(Likelihood):

    def __init__(self, x_dim, sep=4, weights=(0.4, 0.3, 0.2, 0.1), sigma=1):
        assert len(weights) in [2, 3, 4], ('Weights must have 2, 3 or 4 components. Weights=' + str(weights))
        assert np.isclose(sum(weights), 1), ('Weights must sum to 1! Weights=' + str(weights))
        self.sep = sep
        self.weights = weights
        self.sigma = sigma
        self.sigmas = [sigma] * len(weights)
        positions = []
        positions.append(np.asarray([0, sep]))
        positions.append(np.asarray([0, -sep]))
        positions.append(np.asarray([sep, 0]))
        positions.append(np.asarray([-sep, 0]))
        self.positions = positions[:len(weights)]
        super(GaussianMix, self).__init__(x_dim)

    def loglike(self, theta):
        thetas = []
        for pos in self.positions:
            thetas.append(copy.deepcopy(theta))
            thetas[-1][:2] -= pos
        logls = [(log_gaussian_pdf(thetas[i], sigma=self.sigmas[i])
                  + np.log(self.weights[i])) for i in range(len(self.weights))]
        return scipy.special.logsumexp(logls)

    @property
    def max_loglike(self):
        return self(self.positions[np.argmax(self.weights)])

import torch
from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
import numpy as np

from nnest.trainer import Trainer
from nnest.distributions import GeneralisedNormal

max_forward_backward_diff = 1.0E-5
np.random.seed(0)


def test_base_dist():
    for dims in [2, 3, 4, 5]:
        base_dists = [
            TransformedDistribution(Uniform(torch.zeros(dims), torch.ones(dims)), SigmoidTransform().inv),
            MultivariateNormal(torch.zeros(dims), torch.eye(dims)),
            GeneralisedNormal(torch.zeros(dims), torch.ones(dims), torch.tensor(8.0))
        ]
        for base_dist in base_dists:
            t = Trainer(dims, flow='choleksy', base_dist=base_dist)
            test_data = np.random.normal(size=(10, dims))
            test_data = torch.from_numpy(test_data).float()
            z, z_log_det = t.forward(test_data)
            assert z.shape == torch.Size([10, dims])
            assert z_log_det.shape == torch.Size([10])
            x, x_log_det = t.inverse(z)
            diff = torch.max(x - test_data).detach().cpu().numpy()
            assert np.abs(diff) <= max_forward_backward_diff
            diff = torch.max(x_log_det + z_log_det).detach().cpu().numpy()
            assert np.abs(diff) <= max_forward_backward_diff
            samples = t.get_synthetic_samples(10)
            assert samples.shape == torch.Size([10, dims])
            log_probs = t.log_probs(test_data)
            assert log_probs.shape == torch.Size([10])


def test_choleksy():
    for dims in [2, 3, 4, 5]:
        t = Trainer(dims, flow='choleksy')
        test_data = np.random.normal(size=(10, dims))
        test_data = torch.from_numpy(test_data).float()
        z, z_log_det = t.forward(test_data)
        assert z.shape == torch.Size([10, dims])
        assert z_log_det.shape == torch.Size([10])
        x, x_log_det = t.inverse(z)
        diff = torch.max(x - test_data).detach().cpu().numpy()
        assert np.abs(diff) <= max_forward_backward_diff
        diff = torch.max(x_log_det + z_log_det).detach().cpu().numpy()
        assert np.abs(diff) <= max_forward_backward_diff
        samples = t.get_synthetic_samples(10)
        assert samples.shape == torch.Size([10, dims])
        log_probs = t.log_probs(test_data)
        assert log_probs.shape == torch.Size([10])


def test_nvp():
    for dims in [2, 3, 4, 5]:
        t = Trainer(dims, flow='nvp')
        test_data = np.random.normal(size=(10, dims))
        test_data = torch.from_numpy(test_data).float()
        z, z_log_det = t.forward(test_data)
        assert z.shape == torch.Size([10, dims])
        assert z_log_det.shape == torch.Size([10])
        x, x_log_det = t.inverse(z)
        diff = torch.max(x - test_data).detach().cpu().numpy()
        assert np.abs(diff) <= max_forward_backward_diff
        diff = torch.max(x_log_det + z_log_det).detach().cpu().numpy()
        assert np.abs(diff) <= max_forward_backward_diff
        samples = t.get_synthetic_samples(10)
        assert samples.shape == torch.Size([10, dims])
        log_probs = t.log_probs(test_data)
        assert log_probs.shape == torch.Size([10])


def test_spline():
    for dims in [2, 3, 4, 5]:
        t = Trainer(dims, flow='spline')
        test_data = np.random.normal(size=(10, dims))
        test_data = torch.from_numpy(test_data).float()
        z, z_log_det = t.forward(test_data)
        assert z.shape == torch.Size([10, dims])
        assert z_log_det.shape == torch.Size([10])
        x, x_log_det = t.inverse(z)
        diff = torch.max(x - test_data).detach().cpu().numpy()
        assert np.abs(diff) <= max_forward_backward_diff
        diff = torch.max(x_log_det + z_log_det).detach().cpu().numpy()
        assert np.abs(diff) <= max_forward_backward_diff
        samples = t.get_synthetic_samples(10)
        assert samples.shape == torch.Size([10, dims])
        log_probs = t.log_probs(test_data)
        assert log_probs.shape == torch.Size([10])


def test_nvp_slow():
    for num_slow in [2, 3, 4, 5]:
        for num_fast in [2, 3, 4, 5]:
            dims = num_slow + num_fast
            t = Trainer(dims, num_slow=num_slow, flow='nvp')
            test_data = np.random.normal(size=(10, dims))
            test_data = torch.from_numpy(test_data).float()
            z, z_log_det = t.forward(test_data)
            assert z.shape == torch.Size([10, dims])
            assert z_log_det.shape == torch.Size([10])
            x, x_log_det = t.inverse(z)
            diff = torch.max(x - test_data).detach().cpu().numpy()
            assert np.abs(diff) <= max_forward_backward_diff
            diff = torch.max(x_log_det + z_log_det).detach().cpu().numpy()
            assert np.abs(diff) <= max_forward_backward_diff
            dz = torch.randn_like(z) * 0.01
            dz[:, 0:num_slow] = 0.0
            xp, log_det = t.inverse(z + dz)
            diff = torch.max((x - xp)[:, :num_slow]).detach().cpu().numpy()
            assert np.abs(diff) == 0
            samples = t.get_synthetic_samples(10)
            assert samples.shape == torch.Size([10, dims])
            log_probs = t.log_probs(test_data)
            assert log_probs.shape == torch.Size([10])


def test_spline_slow():
    for num_slow in [2, 3, 4, 5]:
        for num_fast in [2, 3, 4, 5]:
            dims = num_slow + num_fast
            t = Trainer(dims, num_slow=num_slow, flow='spline')
            test_data = np.random.normal(size=(10, dims))
            test_data = torch.from_numpy(test_data).float()
            z, z_log_det = t.forward(test_data)
            assert z.shape == torch.Size([10, dims])
            assert z_log_det.shape == torch.Size([10])
            x, x_log_det = t.inverse(z)
            diff = torch.max(x - test_data).detach().cpu().numpy()
            assert np.abs(diff) <= max_forward_backward_diff
            diff = torch.max(x_log_det + z_log_det).detach().cpu().numpy()
            assert np.abs(diff) <= max_forward_backward_diff
            dz = torch.randn_like(z) * 0.01
            dz[:, 0:num_slow] = 0.0
            xp, log_det = t.inverse(z + dz)
            diff = torch.max((x - xp)[:, :num_slow]).detach().cpu().numpy()
            assert np.abs(diff) == 0
            samples = t.get_synthetic_samples(10)
            assert samples.shape == torch.Size([10, dims])
            log_probs = t.log_probs(test_data)
            assert log_probs.shape == torch.Size([10])

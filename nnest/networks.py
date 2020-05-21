"""
.. module:: networks
   :synopsis: Flow neural networks for single speed and fast-slow hierarchies
.. moduleauthor:: Adam Moss <adam.moss@nottingham.ac.uk>
"""

import itertools

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import distributions
import numpy as np


class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super(NormalizingFlow, self).__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
            zs.append(x)
        return zs[-1], log_det

    def inverse(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z)
            log_det += ld
            xs.append(z)
        return xs[-1], log_det


class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """

    def __init__(self, num_inputs, flows, prior=None, device=None):
        super(NormalizingFlowModel, self).__init__()
        self.num_inputs = num_inputs
        if prior is None:
            if device is not None:
                self.prior = distributions.MultivariateNormal(torch.zeros(num_inputs).to(device),
                                                              torch.eye(num_inputs).to(device))
            else:
                self.prior = distributions.MultivariateNormal(torch.zeros(num_inputs),
                                                              torch.eye(num_inputs))
        else:
            self.prior = prior
        self.flow = NormalizingFlow(flows)
        if device is not None:
            self.flow.to(device)
        self.device = device

    def forward(self, x):
        return self.flow.forward(x)

    def inverse(self, z):
        return self.flow.inverse(z)

    def log_probs(self, inputs):
        u, log_det = self.forward(inputs)
        log_probs = self.prior.log_prob(u)
        return log_probs + log_det

    def sample(self, num_samples=None, noise=None):
        if noise is None:
            noise = self.prior.sample((num_samples,))
        if self.device is not None:
            noise = noise.to(self.device)
        samples, _ = self.inverse(noise)
        return samples


class FastSlowNormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """

    def __init__(self, num_fast, num_slow, fast_flows, slow_flows, prior=None, device=None):
        super(FastSlowNormalizingFlowModel, self).__init__()
        self.num_fast = num_fast
        self.num_slow = num_slow
        self.num_inputs = num_slow + num_fast
        if prior is None:
            if device is not None:
                self.prior = distributions.MultivariateNormal(torch.zeros(self.num_inputs).to(device),
                                                              torch.eye(self.num_inputs).to(device))
            else:
                self.prior = distributions.MultivariateNormal(torch.zeros(self.num_inputs),
                                                              torch.eye(self.num_inputs))
        else:
            self.prior = prior
        self.fast_flow = NormalizingFlow(fast_flows)
        self.slow_flow = NormalizingFlow(slow_flows)
        if device is not None:
            self.fast_flow.to(device)
            self.slow_flow.to(device)
        # Combine fast and slow such that slow is unnchanged just by updating fast block
        mask = torch.cat((torch.ones(num_slow), torch.zeros(num_fast)))
        if device is not None:
            mask = mask.to(device)
        flows = [
            CouplingLayer(
                num_slow + num_fast, 64, mask,
                s_act='tanh', t_act='relu', num_layers=1)
        ]
        self.flow = NormalizingFlow(flows)
        if device is not None:
            self.flow.to(device)
        self.device = device

    def forward(self, inputs):
        slow, logdets_slow = self.slow_flow.forward(inputs[:, :self.num_slow])
        fast, logdets_fast = self.fast_flow.forward(inputs[:, self.num_slow:])
        inputs = torch.cat((slow, fast), dim=1)
        inputs, logdets = self.flow.forward(inputs)
        return inputs, logdets_slow + logdets_fast + logdets

    def inverse(self, inputs):
        inputs, logdets = self.flow.inverse(inputs)
        slow, logdets_slow = self.slow_flow.inverse(inputs[:, :self.num_slow])
        fast, logdets_fast = self.fast_flow.inverse(inputs[:, self.num_slow:])
        inputs = torch.cat((slow, fast), dim=1)
        return inputs, logdets_slow + logdets_fast + logdets

    def log_probs(self, inputs):
        slow, logdets_slow = self.slow_flow.forward(inputs[:, :self.num_slow])
        fast, logdets_fast = self.fast_flow.forward(inputs[:, self.num_slow:])
        inputs = torch.cat((slow, fast), dim=1)
        u, log_det = self.flow.forward(inputs)
        log_probs = self.prior.log_prob(u)
        return log_probs + log_det + logdets_slow + logdets_fast

    def sample(self, num_samples=None, noise=None):
        if noise is None:
            noise = self.prior.sample((num_samples,))
        if self.device is not None:
            noise = noise.to(self.device)
        samples, _ = self.inverse(noise)
        return samples


"""  RealNVP    
Coupling layer from https://github.com/ikostrikov/pytorch-flows modified 
to use variable number of layers
"""


class CouplingLayer(nn.Module):
    """ An implementation of a coupling layer
    from RealNVP (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 mask,
                 s_act='tanh',
                 t_act='relu',
                 num_layers=2,
                 translate_only=False):
        super(CouplingLayer, self).__init__()

        self.num_inputs = num_inputs
        self.mask = mask
        self.translate_only = translate_only

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        s_act_func = activations[s_act]
        t_act_func = activations[t_act]

        if not translate_only:
            scale_layers = [nn.Linear(num_inputs, num_hidden), s_act_func()]
            for i in range(0, num_layers):
                scale_layers += [nn.Linear(num_hidden, num_hidden), s_act_func()]
            scale_layers += [nn.Linear(num_hidden, num_inputs)]
            self.scale_net = nn.Sequential(*scale_layers)

        translate_layers = [nn.Linear(num_inputs, num_hidden), t_act_func()]
        for i in range(0, num_layers):
            translate_layers += [nn.Linear(num_hidden, num_hidden), t_act_func()]
        translate_layers += [nn.Linear(num_hidden, num_inputs)]
        self.translate_net = nn.Sequential(*translate_layers)

        def init(m):
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(0)
                nn.init.orthogonal_(m.weight.data)

    def forward(self, inputs):
        mask = self.mask
        masked_inputs = inputs * mask
        t = self.translate_net(masked_inputs) * (1 - mask)
        if self.translate_only:
            return inputs + t, 0
        else:
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            s = torch.exp(log_s)
            return inputs * s + t, log_s.sum(-1)

    def inverse(self, inputs):
        mask = self.mask
        masked_inputs = inputs * mask
        t = self.translate_net(masked_inputs) * (1 - mask)
        if self.translate_only:
            return inputs - t, 0
        else:
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            s = torch.exp(-log_s)
            return (inputs - t) * s, -log_s.sum(-1)


class ScaleLayer(nn.Module):

    def __init__(self):
        super(ScaleLayer, self).__init__()

        self.scale = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, inputs):
        s = torch.exp(self.scale)
        return inputs * s, self.scale.sum(-1)

    def inverse(self, inputs):
        s = torch.exp(-self.scale)
        return inputs * s, -self.scale.sum(-1)


class SingleSpeedNVP(NormalizingFlowModel):

    def __init__(self, num_inputs, num_hidden, num_blocks, num_layers, scale='',
                 prior=None, device=None):
        translate_only = scale == 'translate' or scale == 'constant'
        mask = torch.arange(0, num_inputs) % 2
        mask = mask.float()
        if device is not None:
            mask = mask.to(device)
        flows = []
        for _ in range(num_blocks):
            flows += [
                CouplingLayer(
                    num_inputs, num_hidden, mask,
                    s_act='tanh', t_act='relu', num_layers=num_layers, translate_only=translate_only),
            ]
            if scale == 'constant':
                flows += [ScaleLayer()]
            mask = 1 - mask
        super(SingleSpeedNVP, self).__init__(num_inputs, flows, prior=prior, device=device)


class FastSlowNVP(FastSlowNormalizingFlowModel):

    def __init__(self, num_fast, num_slow, num_hidden, num_blocks, num_layers, scale='',
                 prior=None, device=None):
        # Fast block
        mask_fast = torch.arange(0, num_fast) % 2
        mask_fast = mask_fast.float()
        if device is not None:
            mask_fast = mask_fast.to(device)
        fast_flows = []
        for _ in range(num_blocks):
            fast_flows += [
                CouplingLayer(
                    num_fast, num_hidden, mask_fast,
                    s_act='tanh', t_act='relu', num_layers=num_layers)
            ]
            mask_fast = 1 - mask_fast
        # Slow block
        mask_slow = torch.arange(0, num_slow) % 2
        mask_slow = mask_slow.float()
        if device is not None:
            mask_slow = mask_slow.to(device)
        slow_flows = []
        for _ in range(num_blocks):
            slow_flows += [
                CouplingLayer(
                    num_slow, num_hidden, mask_slow,
                    s_act='tanh', t_act='relu', num_layers=num_layers)
            ]
            mask_slow = 1 - mask_slow
        super(FastSlowNVP, self).__init__(num_fast, num_slow, fast_flows, slow_flows, prior=prior, device=device)


"""
Neural Spline Flows

Paper reference: Durkan et al https://arxiv.org/abs/1906.04032
From https://github.com/karpathy/pytorch-normalizing-flows, itself based on 
 https://github.com/tonyduan/normalizing-flows/blob/master/nf/flows.py
 Modified here to include odd numbers of dimensions
"""


class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, nin, nout, nh):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )

    def forward(self, x):
        return self.net(x)


DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1


def unconstrained_RQS(inputs, unnormalized_widths, unnormalized_heights,
                      unnormalized_derivatives, inverse=False,
                      tail_bound=1., min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                      min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                      min_derivative=DEFAULT_MIN_DERIVATIVE):
    inside_intvl_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_intvl_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0

    outputs[inside_intvl_mask], logabsdet[inside_intvl_mask] = RQS(
        inputs=inputs[inside_intvl_mask],
        unnormalized_widths=unnormalized_widths[inside_intvl_mask, :],
        unnormalized_heights=unnormalized_heights[inside_intvl_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_intvl_mask, :],
        inverse=inverse,
        left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative
    )
    return outputs, logabsdet


def RQS(inputs, unnormalized_widths, unnormalized_heights,
        unnormalized_derivatives, inverse=False, left=0., right=1.,
        bottom=0., top=1., min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError("Input outside domain")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)
    input_derivatives_plus_one = input_derivatives_plus_one[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (((inputs - input_cumheights) * (input_derivatives \
                                             + input_derivatives_plus_one - 2 * input_delta) \
              + input_heights * (input_delta - input_derivatives)))
        b = (input_heights * input_derivatives - (inputs - input_cumheights) \
             * (input_derivatives + input_derivatives_plus_one \
                - 2 * input_delta))
        c = - input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta \
                      + ((input_derivatives + input_derivatives_plus_one \
                          - 2 * input_delta) * theta_one_minus_theta)
        derivative_numerator = input_delta.pow(2) \
                               * (input_derivatives_plus_one * root.pow(2) \
                                  + 2 * input_delta * theta_one_minus_theta \
                                  + input_derivatives * (1 - root).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2) \
                                     + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives \
                                      + input_derivatives_plus_one - 2 * input_delta) \
                                     * theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) \
                               * (input_derivatives_plus_one * theta.pow(2) \
                                  + 2 * input_delta * theta_one_minus_theta \
                                  + input_derivatives * (1 - theta).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, logabsdet


class NSF_CL(nn.Module):
    """ Neural spline flow, coupling layer, [Durkan et al. 2019] """

    def __init__(self, dim, K=5, B=3, hidden_dim=8, base_network=MLP):
        super(NSF_CL, self).__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.even = self.dim == 2 * self.half_dim
        self.K = K
        self.B = B
        if self.even:
            self.f1 = base_network(self.half_dim, (3 * K - 1) * self.half_dim, hidden_dim)
            self.f2 = base_network(self.half_dim, (3 * K - 1) * self.half_dim, hidden_dim)
        else:
            self.f1 = base_network(self.half_dim + 1, (3 * K - 1) * self.half_dim, hidden_dim)
            self.f2 = base_network(self.half_dim, (3 * K - 1) * (self.half_dim + 1), hidden_dim)

    def forward(self, x):
        log_det = torch.zeros(x.shape[0])
        if self.even:
            lower, upper = x[:, :self.half_dim], x[:, self.half_dim:]
        else:
            lower, upper = x[:, :self.half_dim + 1], x[:, self.half_dim + 1:]
        out = self.f1(lower).reshape(-1, self.half_dim, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim=2)
        W, H = torch.softmax(W, dim=2), torch.softmax(H, dim=2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        upper, ld = unconstrained_RQS(upper, W, H, D, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld, dim=1)
        if self.even:
            out = self.f2(upper).reshape(-1, self.half_dim, 3 * self.K - 1)
        else:
            out = self.f2(upper).reshape(-1, self.half_dim + 1, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim=2)
        W, H = torch.softmax(W, dim=2), torch.softmax(H, dim=2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        lower, ld = unconstrained_RQS(lower, W, H, D, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld, dim=1)
        return torch.cat([lower, upper], dim=1), log_det

    def inverse(self, z):
        log_det = torch.zeros(z.shape[0])
        if self.even:
            lower, upper = z[:, :self.half_dim], z[:, self.half_dim:]
            out = self.f2(upper).reshape(-1, self.half_dim, 3 * self.K - 1)
        else:
            lower, upper = z[:, :self.half_dim + 1], z[:, self.half_dim + 1:]
            out = self.f2(upper).reshape(-1, self.half_dim + 1, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim=2)
        W, H = torch.softmax(W, dim=2), torch.softmax(H, dim=2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        lower, ld = unconstrained_RQS(lower, W, H, D, inverse=True, tail_bound=self.B)
        log_det += torch.sum(ld, dim=1)
        out = self.f1(lower).reshape(-1, self.half_dim, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim=2)
        W, H = torch.softmax(W, dim=2), torch.softmax(H, dim=2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        upper, ld = unconstrained_RQS(upper, W, H, D, inverse=True, tail_bound=self.B)
        log_det += torch.sum(ld, dim=1)
        return torch.cat([lower, upper], dim=1), log_det


class Invertible1x1Conv(nn.Module):
    """
    As introduced in Glow paper.
    """

    def __init__(self, dim):
        super(Invertible1x1Conv, self).__init__()
        self.dim = dim
        Q = torch.nn.init.orthogonal_(torch.randn(dim, dim))
        P, L, U = torch.lu_unpack(*Q.lu())
        self.P = P  # remains fixed during optimization
        self.L = nn.Parameter(L)  # lower triangular portion
        self.S = nn.Parameter(U.diag())  # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(torch.triu(U, diagonal=1))  # "crop out" diagonal, stored in S

    def _assemble_W(self):
        """ assemble W from its pieces (P, L, U, S) """
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim))
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def forward(self, x):
        W = self._assemble_W()
        z = x @ W
        log_det = torch.sum(torch.log(torch.abs(self.S)))
        return z, log_det

    def inverse(self, z):
        W = self._assemble_W()
        W_inv = torch.inverse(W)
        x = z @ W_inv
        log_det = -torch.sum(torch.log(torch.abs(self.S)))
        return x, log_det


class AffineConstantFlow(nn.Module):
    """
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is None
    """

    def __init__(self, dim, scale=True, shift=True):
        super(AffineConstantFlow, self).__init__()
        self.s = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if scale else None
        self.t = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if shift else None

    def forward(self, x):
        s = self.s if self.s is not None else x.new_zeros(x.size())
        t = self.t if self.t is not None else x.new_zeros(x.size())
        z = x * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def inverse(self, z):
        s = self.s if self.s is not None else z.new_zeros(z.size())
        t = self.t if self.t is not None else z.new_zeros(z.size())
        x = (z - t) * torch.exp(-s)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class ActNorm(AffineConstantFlow):
    """
    Really an AffineConstantFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """

    def __init__(self, *args, **kwargs):
        super(ActNorm, self).__init__(*args, **kwargs)
        self.data_dep_init_done = False

    def forward(self, x):
        # first batch is used for init
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None  # for now
            self.s.data = (-torch.log(x.std(dim=0, keepdim=True))).detach()
            self.t.data = (-(x * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
            self.data_dep_init_done = True
        return super(ActNorm, self).forward(x)


class SingleSpeedSpline(NormalizingFlowModel):

    def __init__(self, num_inputs, hidden_dim, num_blocks, num_bins=8, tail_bound=3, prior=None, device=None):
        flows = [NSF_CL(dim=num_inputs, K=num_bins, B=tail_bound, hidden_dim=hidden_dim) for _ in range(num_blocks)]
        convs = [Invertible1x1Conv(dim=num_inputs) for _ in flows]
        norms = [ActNorm(dim=num_inputs) for _ in flows]
        flows = list(itertools.chain(*zip(norms, convs, flows)))
        super(SingleSpeedSpline, self).__init__(num_inputs, flows, prior=prior, device=device)


class FastSlowSpline(FastSlowNormalizingFlowModel):

    def __init__(self, num_fast, num_slow, hidden_dim, num_blocks, num_bins=8, tail_bound=3, prior=None, device=None):
        # Fast block
        fast_flows = [NSF_CL(dim=num_fast, K=num_bins, B=tail_bound, hidden_dim=16) for _ in range(num_blocks)]
        convs = [Invertible1x1Conv(dim=num_fast) for _ in fast_flows]
        norms = [ActNorm(dim=num_fast) for _ in fast_flows]
        fast_flows = list(itertools.chain(*zip(norms, convs, fast_flows)))
        # Slow block
        slow_flows = [NSF_CL(dim=num_slow, K=num_bins, B=tail_bound, hidden_dim=hidden_dim) for _ in range(num_blocks)]
        convs = [Invertible1x1Conv(dim=num_slow) for _ in slow_flows]
        norms = [ActNorm(dim=num_slow) for _ in slow_flows]
        slow_flows = list(itertools.chain(*zip(norms, convs, slow_flows)))
        super(FastSlowSpline, self).__init__(num_fast, num_slow, fast_flows, slow_flows, prior=prior, device=device)

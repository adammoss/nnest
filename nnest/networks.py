"""
.. module:: networks
   :synopsis: Flow neural networks for single speed and fast-slow hierarchies
.. moduleauthor:: Adam Moss <adam.moss@nottingham.ac.uk>
Coupling layer from https://github.com/ikostrikov/pytorch-flows modified 
to use variable number of layers
"""

import math

import torch
import torch.nn as nn


class BatchNormFlow(nn.Module):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (
                    inputs - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data *
                                       (1 - self.momentum))
                self.running_var.add_(self.batch_var.data *
                                      (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(
                -1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(
                -1, keepdim=True)


class CouplingLayer(nn.Module):
    """ An implementation of a coupling layer
    from RealNVP (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 mask,
                 num_cond_inputs=None,
                 s_act='tanh',
                 t_act='relu',
                 num_layers=2,
                 device=None):
        super(CouplingLayer, self).__init__()

        self.num_inputs = num_inputs
        self.mask = mask

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        s_act_func = activations[s_act]
        t_act_func = activations[t_act]

        if num_cond_inputs is not None:
            total_inputs = num_inputs + num_cond_inputs
        else:
            total_inputs = num_inputs
        
        scale_layers = [nn.Linear(total_inputs, num_hidden), s_act_func()]
        for i in range(0, num_layers):
            scale_layers += [nn.Linear(num_hidden, num_hidden), s_act_func()]
        scale_layers += [nn.Linear(num_hidden, num_inputs)]
        self.scale_net = nn.Sequential(*scale_layers)
        translate_layers = [nn.Linear(total_inputs, num_hidden), t_act_func()]
        for i in range(0, num_layers):
            translate_layers += [nn.Linear(num_hidden, num_hidden), t_act_func()]
        translate_layers += [nn.Linear(num_hidden, num_inputs)]
        self.translate_net = nn.Sequential(*translate_layers)
            
        def init(m):
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(0)
                nn.init.orthogonal_(m.weight.data)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        mask = self.mask

        masked_inputs = inputs * mask
        if cond_inputs is not None:
            masked_inputs = torch.cat([masked_inputs, cond_inputs], -1)
        
        if mode == 'direct':
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            s = torch.exp(log_s)
            return inputs * s + t, log_s.sum(-1, keepdim=True)
        else:
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            s = torch.exp(-log_s)
            return (inputs - t) * s, -log_s.sum(-1, keepdim=True)


class FlowSequential(nn.Sequential):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet

        return inputs, logdets

    def log_probs(self, inputs, cond_inputs=None):
        u, log_jacob = self(inputs, cond_inputs)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)
        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.forward(noise, cond_inputs, mode='inverse')[0]
        return samples


class SingleSpeed(nn.Module):

    def __init__(self, num_inputs, num_hidden, num_blocks, num_layers, device=None):
        super(SingleSpeed, self).__init__()
        mask = torch.arange(0, num_inputs) % 2
        mask = mask.float()
        if device is not None:
            mask = mask.to(device)
        modules = []
        for _ in range(num_blocks):
            modules += [
                CouplingLayer(
                    num_inputs, num_hidden, mask, None,
                    s_act='tanh', t_act='relu', num_layers=num_layers)
            ]
            mask = 1 - mask
        self.net = FlowSequential(*modules)
        if device is not None:
            self.net.to(device)

    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
        return self.net(inputs, cond_inputs=cond_inputs, mode=mode, logdets=logdets)

    def log_probs(self, inputs, cond_inputs=None):
        return self.net.log_probs(inputs, cond_inputs=None)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        return self.net.sample(num_samples=num_samples, noise=noise, cond_inputs=cond_inputs)


class FastSlow(nn.Module):

    def __init__(self, num_fast, num_slow, num_hidden, num_blocks, num_layers, device=None):
        super(FastSlow, self).__init__()

        self.num_fast = num_fast
        self.num_slow = num_slow
        self.num_inputs = num_fast + num_slow

        # Fast block
        mask_fast = torch.arange(0, num_fast) % 2
        mask_fast = mask_fast.float()
        if device is not None:
            mask_fast = mask_fast.to(device)
        modules_fast = []
        for _ in range(num_blocks):
            modules_fast += [
                CouplingLayer(
                    num_fast, num_hidden, mask_fast, None,
                    s_act='tanh', t_act='relu', num_layers=num_layers)
            ]
            mask_fast = 1 - mask_fast
        self.net_fast = FlowSequential(*modules_fast)

        # Slow block
        mask_slow = torch.arange(0, num_slow) % 2
        mask_slow = mask_slow.float()
        if device is not None:
            mask_slow = mask_slow.to(device)
        modules_slow = []
        for _ in range(num_blocks):
            modules_slow += [
                CouplingLayer(
                    num_slow, num_hidden, mask_slow, None,
                    s_act='tanh', t_act='relu', num_layers=num_layers)
            ]
            mask_slow = 1 - mask_slow
        self.net_slow = FlowSequential(*modules_slow)

        # Combine fast and slow such that slow is unnchanged just by updating fast block
        modules = []
        mask = torch.cat((torch.ones(num_slow), torch.zeros(num_fast)))
        if device is not None:
            mask = mask.to(device)
        modules = [
            CouplingLayer(
                num_slow + num_fast, num_hidden, mask, None,
                s_act='tanh', t_act='relu', num_layers=num_layers)
        ]
        self.net = FlowSequential(*modules)

    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            slow, logdets_slow = self.net_slow(inputs[:, 0:self.num_slow], mode=mode)
            fast, logdets_fast = self.net_fast(inputs[:, self.num_slow:self.num_slow+self.num_fast], mode=mode)
            inputs = torch.cat((slow, fast), dim=1)
            inputs, logdets = self.net(inputs, mode=mode)
        else:
            inputs, logdets = self.net(inputs, mode=mode)
            slow, logdets_slow = self.net_slow(inputs[:, 0:self.num_slow], mode=mode)
            fast, logdets_fast = self.net_fast(inputs[:, self.num_slow:self.num_slow+self.num_fast], mode=mode)
            inputs = torch.cat((slow, fast), dim=1)
        return inputs, logdets_slow + logdets_fast + logdets

    def log_probs(self, inputs, cond_inputs=None):
        slow, logdets_slow = self.net_slow(inputs[:, 0:self.num_slow])
        fast, logdets_fast = self.net_fast(inputs[:, self.num_slow:self.num_slow+self.num_fast])
        inputs = torch.cat((slow, fast), dim=1)
        u, log_jacob = self.net(inputs)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)
        return (log_probs + log_jacob + logdets_slow + logdets_fast).sum(-1, keepdim=True)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.forward(noise, cond_inputs, mode='inverse')[0]
        return samples

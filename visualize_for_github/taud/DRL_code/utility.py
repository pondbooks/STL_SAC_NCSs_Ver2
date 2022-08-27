import math
import torch

def calculate_log_pi(log_stds, noises, actions):

    gaussian_log_probs = \
        (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)

    # Revised for tanh
    log_pis = gaussian_log_probs - torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

    return log_pis

def reparameterize(means, log_stds):
    """ Reparameterization Trick """
    # standard deviation．
    stds = log_stds.exp()
    # Sampling noises from standard normal distribution. 
    noises = torch.randn_like(means)
    # Compute samples from N(means, stds) using Reparameterization Trick．
    us = means + noises * stds
    # tanh　
    actions = torch.tanh(us)

    # Calculate the logarithm of the probability density of stochastic action．
    log_pis = calculate_log_pi(log_stds, noises, actions)

    return actions, log_pis

import torch
from functools import partial


def get_examples(src, mu=-1.5, **kwargs):
    # select source function
    if src == 'example1':
        source_func = partial(gaussian_like, mu=mu)
    elif src == 'example1_grad':
        source_func = partial(gaussian_gradient, mu=mu)
    elif src == 'example2':
        source_func = partial(gaussian_sum, mu=[-1.5, -1.0, -0.5], sigma=[0.1, 0.2, 0.1])
    elif src == 'example2_grad':
        source_func = partial(gaussian_sum_gradient, mu=[-1.5, -1.0, -0.5], sigma=[0.1, 0.2, 0.1])
    else:
        raise NotImplementedError
    return source_func

def gaussian_like(x, mu=0, sigma=0.1):
    """normalized gaussian distribution"""
    return torch.exp(-0.5 * (x - mu) ** 2 / (sigma ** 2))

def gaussian_gradient(x, mu=0, sigma=0.1):
    """Gradient of the normalized gaussian distribution with respect to x"""
    return -((x - mu) * torch.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)) / sigma ** 2

def gaussian_sum(x, mu=[0], sigma=[0.1]):
    """Sum of normalized gaussian distributions"""
    assert len(mu) == len(sigma)
    return sum([gaussian_like(x, mu[i], sigma[i]) for i in range(len(mu))])

def gaussian_sum_gradient(x, mu=[0], sigma=[0.1]):
    """Gradient of the sum of normalized gaussian distributions with respect to x"""
    assert len(mu) == len(sigma)
    return sum([gaussian_gradient(x, mu[i], sigma[i]) for i in range(len(mu))])
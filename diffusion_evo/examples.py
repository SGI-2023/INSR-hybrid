import torch
from functools import partial


def get_examples(src, mu=0, **kwargs):
    # select source function
    if src == 'example1':
        source_func = partial(gaussian_like, mu=mu)
    elif src == 'example1_lap':
        source_func = partial(gaussian_gradient, mu=mu)
    else:
        raise NotImplementedError
    return source_func

def gaussian_like(x, mu=0, sigma=0.1):
    """normalized gaussian distribution"""
    return torch.exp(-0.5 * (x - mu) ** 2 / (sigma ** 2))

def gaussian_gradient(x, mu=0, sigma=0.1):
    """Gradient of the normalized gaussian distribution with respect to x"""
    return -((x - mu) * torch.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)) / sigma ** 2

def laplacian_gaussian(x, mu=0, sigma=0.1):
    """Laplacian of the normalized gaussian distribution"""
    term1 = torch.exp(-0.5 * (x - mu) ** 2 / sigma ** 2) / sigma ** 2
    term2 = ((x - mu) ** 2 / sigma ** 2) - 1
    return term1 * term2

def gaussian_like_laplacian(x, mu=0, sigma=0.1):
    """Laplacian of the normalized gaussian distribution"""
    return (torch.exp(-0.5 * (x - mu) ** 2 / (sigma ** 2)) * ((x - mu) ** 2 - sigma ** 2) / (sigma ** 4))

# FROM Q's code: https://github.com/qihongl/autor-fn-text/blob/main/src/stats.py

import torch
import numpy as np
from scipy.stats import sem, norm

def compute_stats(arr, axis=0, n_se=2, use_se=True):
    """compute mean and errorbar w.r.t to SE
    Parameters
    ----------
    arr : nd array
        data
    axis : int
        the axis to do stats along with
    n_se : int
        number of SEs
    Returns
    -------
    (n-1)d array, (n-1)d array
        mean and se
    """
    mu_ = np.mean(arr, axis=axis)
    if use_se:
        er_ = sem(arr, axis=axis) * n_se
    else:
        er_ = np.std(arr, axis=axis)
    return mu_, er_

def moving_average(x, winsize):
    return np.convolve(x, np.ones(winsize), 'valid') / winsize

def entropy(probs, to_probs=False, smallnum=1e-9):
    """calculate entropy.
    I'm using log base 2!
    Parameters
    ----------
    probs : a torch vector
        a prob distribution
    Returns
    -------
    torch scalar
        the entropy of the distribution
    """
    if to_probs:
        probs = to_prob_distribution(probs)
    probs = torch.clamp(probs, min=smallnum, max=1-smallnum)
    return - torch.stack([pi * torch.log2(pi) for pi in probs]).sum()

def to_prob_distribution(probs):
    return probs / probs.sum()

def compute_d_prime(signal, noise):
    # Calculate means and standard deviations
    mu_signal, sigma_signal = np.mean(signal), np.std(signal)
    mu_noise, sigma_noise = np.mean(noise), np.std(noise)
    # Compute d'
    d_prime = (mu_signal - mu_noise) / np.sqrt(0.5 * (sigma_signal**2 + sigma_noise**2))
    return d_prime



if __name__ == "__main__":
    d1 = torch.Tensor([.5, .5])
    d2 = torch.Tensor([1, 0])

    print(entropy(d1))
    print(entropy(d2))
# FROM Q: https://github.com/qihongl/autor-fn-text/blob/main/src/utils.py

import pickle

def pickle_save_dict(input_dict, save_path):
    """Save the dictionary"""
    with open(save_path, 'wb') as handle:
        pickle.dump(input_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load_dict(fpath):
    """Load the dict"""
    return pickle.load(open(fpath, "rb"))

# FROM Q: https://github.com/qihongl/autor-fn-text/blob/main/src/stats.py

import numpy as np
from scipy.stats import sem

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

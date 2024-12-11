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
    mu_ = np.nanmean(arr, axis=axis)
    if use_se:
        er_ = sem(arr, axis=axis, nan_policy='omit') * n_se
    else:
        er_ = np.nanstd(arr, axis=axis)
    return mu_, er_


#####  my own
from datetime import datetime, timedelta

# wrappers for datetime.strftime and strptime for our specific format

def string_to_datetime(string, pattern = "%Y%m%d_%H%M_%S.%f"):
    return datetime.strptime(string, pattern)

def datetime_to_string(date, pattern ="%Y%m%d_%H%M_%S.%f", truncate_digits=None):
    if truncate_digits:
        return datetime.strftime(date, pattern)[:-truncate_digits]
    return datetime.strftime(date, pattern)

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
from scipy.optimize import curve_fit

# wrappers for datetime.strftime and strptime for our specific format

def string_to_datetime(string, pattern = "%Y%m%d_%H%M_%S.%f"):
    return datetime.strptime(string, pattern)

def datetime_to_string(date, pattern ="%Y%m%d_%H%M_%S.%f", truncate_digits=None):
    if truncate_digits:
        return datetime.strftime(date, pattern)[:-truncate_digits]
    return datetime.strftime(date, pattern)

# fitting power law and exponential curves
def fit_acf(acf_data, model):
    if model.lower() == 'power law':
        fit_function = lambda x,a,b: a * x**(b)
    elif model.lower() == 'truncated power law':
        fit_function = lambda x,a,b: x**a * np.exp(b * x) 
    elif model.lower() == 'exponential':
        fit_function = lambda x,a,b: a * np.exp(b * x) 
    elif model.lower() == 'stretched_exponential':
        fit_function = lambda x,a,b: np.exp(a * x**b)
    elif model.lower() == 'lognormal':
        fit_function = lambda x,a,b: np.exp(a * x**b)

    x = np.arange(1, len(acf_data)+1)
    params = curve_fit(fit_function, x, acf_data)
    return fit_function(x, *params)
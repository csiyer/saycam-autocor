# FROM Q: https://github.com/qihongl/autor-fn-text/blob/main/src/stats.py

import pickle

def pickle_save_dict(input_dict, save_path):
    """Save the dictionary"""
    with open(save_path, 'wb') as handle:
        pickle.dump(input_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load_dict(fpath):
    """Load the dict"""
    return pickle.load(open(fpath, "rb"))

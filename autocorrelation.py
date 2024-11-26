"""Functions to calculate autocorrelation measures on model embeddings"""

import os, glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
import statsmodels.api as sm
from joblib import Parallel, delayed
from utils import compute_stats, pickle_load_dict, pickle_save_dict

sns.set(style='white', palette='colorblind', context='talk')
plt.rcParams['figure.dpi'] = 100


def get_consec_dists(all_embeddings, save_folder=None, plot=True):
    """ 
    gets consecutive distances between pairs of points
    works for either one embedding or a list of them
    """
    if not type(all_embeddings) == list:
        all_embeddings = [all_embeddings]

    gaps = [2 ** k for k in range(9)] # change to range(12) when we have concatenated videos

    consec_dist = {'mu' : [], 'se' : []}

    for embeddings in tqdm(all_embeddings):
        n = embeddings.shape[0] # change to 50000 or something big # how far back to look

        norms_mu = {g: None for g in gaps}
        norms_se = {g: None for g in gaps}
        
        for gap in gaps:
            norms = []
            for i in range(n - gap - 1 ):
                j = i + gap 
                norms.append(np.linalg.norm(embeddings[j] - embeddings[i])) # convert format and concatenate

            norms_mu[gap], norms_se[gap] = compute_stats(norms)
            
        consec_dist['mu'].append(norms_mu)
        consec_dist['se'].append(norms_se)

    if save_folder:
        out_dir = os.path.join(save_folder, 'consec_dist')
        os.makedirs(out_dir, exist_ok=True)
        fpath = os.path.join(out_dir, 'consec_dist')
        pickle_save_dict(consec_dist, fpath+'.pkl')

    if plot:
        f, ax = plt.subplots(1, 1, figsize=(5, 4))
        f.suptitle('average pairwise Euclidean distance by gap')
        for i, (norms_mu, norms_se) in enumerate(zip(consec_dist['mu'], consec_dist['se'])):
            mu = list(norms_mu.values())
            se = list(norms_se.values())
            ax.plot(mu, label=f'video {i}')
            ax.fill_between(
                range(len(mu)), 
                np.array(mu) - np.array(se), 
                np.array(mu) + np.array(se), 
                color='gray', alpha=0.2, zorder=-99, label=f'Uncertainty' if i == 0 else None
            )

        ax.set_xticks(range(len(gaps)))  # Adjust based on the length of `mu`
        # ax.set_xticklabels(gaps, rotation=90)  # Replace `gaps` with your labels if available
        ax.set_xlabel('Sentence distance')
        ax.set_ylabel('Embedding distance')
        # ax.legend()
        sns.despine()
        f.tight_layout()

        if save_folder:
            plt.savefig(fpath+'.png')

        plt.show()

    return consec_dist


def compute_acf_across_dims(embeddings, nlags, perm=None):
    if len(embeddings.shape) < 2:
        embeddings = embeddings[:,np.newaxis]
    dim = embeddings.shape[1]
    if perm is not None:
        embeddings = embeddings[perm]
    acf_ndim_perm = np.array([sm.tsa.acf(embeddings[:, d], nlags=nlags) for d in range(dim)])
    return acf_ndim_perm.mean(axis=0)


# main function
def run_plot_acf(all_embeddings,  n=None, nlags=None, permute=False, n_jobs=1, plot=True, save_folder=None, save_tag = ''):
    """
    Calculates autocorrelation of data and plots!
    Inputs:
        - all_embeddings: list of np.array embeddings of length n_videos and each element shape (timepoints, dims) 
                For the familiarity-novelty timeseries, dims=1 -- still works!
        - n: number of timepoints to consider
        - nlags: lags of the autocorrelation function
        - permute: compute a permuted null by shuffling and recomputing ACF
        - n_jobs: number of jobs to parallelize across
        - bool to plot
        - save_folder: if provided, save outputs and plots there
    This works for both the raw data and familiarity timeseries
    """
    if not type(all_embeddings) == list:
        all_embeddings = [all_embeddings] # make this compatible with a single continuous embeddings file
    
    if all_embeddings[0].shape[1] > 1:
        print('Computing autocorrelation of raw embeddings...')
        plot_title = f'Autocorrelation of raw embeddings (averaged across units), ({save_tag})'
        save_folder_addition = 'acf_raw'
    else:
        print('Computing autocorrelation of familiarity timeseries...')
        plot_title = f'Autocorrelation of familiarity timeseries, ({save_tag})' 
        save_folder_addition = 'acf_fn'
        
    acfs_all = []
    acfs_perm_mu_se_all = []

    for embeddings in all_embeddings: 
        if not n or n > embeddings.shape[0]:
            n = embeddings.shape[0]
        if not nlags or nlags > n:
            nlags = n // 2 # reasonable default?

        embeddings = embeddings[:n] # now (n x dim)

        # calculate autocorrelation of each model unit and average
        acf = compute_acf_across_dims(embeddings, nlags)
        acfs_all.append(acf)

        if permute:
            # permute the embedding units and compute autocrrelation 
            n_iter = 10
            acf_perm = np.array(
                Parallel(n_jobs=n_jobs)(
                    delayed(compute_acf_across_dims)(embeddings, nlags, perm=np.random.permutation(n)) for _ in range(n_iter)
                )
            )
            acf_perm_mu, acf_perm_se = compute_stats(acf_perm, axis=0)
            # w = 10
            # acf_perm_mu_smoothed = moving_average(acf_perm_mu, w)
            acfs_perm_mu_se_all.append((acf_perm_mu, acf_perm_se))

    if save_folder:
        acf_out_dir = os.path.join(save_folder, save_folder_addition)
        os.makedirs(acf_out_dir, exist_ok=True)
        fpath = os.path.join(acf_out_dir, 'acfs_all')
        if save_tag:
            fpath += '_' + save_tag
        pickle_save_dict({'acfs_all': acfs_all, 'acfs_perm_mu_se_all': acfs_perm_mu_se_all}, fpath+'.pkl')
        
    if plot:
        f, ax = plt.subplots(1, 1, figsize=(8, 6))
        f.suptitle(plot_title)
        for i,acf in enumerate(acfs_all):
            # Plot the ACF for the current array
            ax.plot(acf[1:]) 
            if permute:
                # Plot the permuted null mean with shaded SE
                acf_perm_mu, acf_perm_se = acfs_perm_mu_se_all[i]
                ax.fill_between(range(len(acf_perm_mu)), 
                            acf_perm_mu - acf_perm_se, 
                            acf_perm_mu + acf_perm_se, 
                            color='gray', alpha=0.2, zorder=-99, label=f'Permuted null' if i == 0 else None)
                
        ax.set_xlabel('lag')
        ax.set_ylabel('autocorrelation')
        ax.set_xscale('log')
        ax.set_yscale('log')
        if permute: ax.legend()
        sns.despine()
        f.tight_layout()

        if save_folder:
            plt.savefig(fpath+'.png')
        
        plt.show()
    
    return acfs_all, acfs_perm_mu_se_all


def compute_dist_mats(all_embeddings,  n_jobs):
    """Euclidean distance of each pair of timepoints"""
    if n_jobs > 1:
        return Parallel(n_jobs=n_jobs)(
            delayed(lambda e: distance.squareform(distance.pdist(e, 'euclidean')))(e) for e in all_embeddings
        )
    else:
        return [distance.squareform(distance.pdist(e, 'euclidean')) for e in all_embeddings]
    

def compute_percent_familiar(dist_mat, familiar_rad, window_start=None, window_end=None, n=None):
    """
    Computes the proportion of distances within a familiar radius for a sliding window.
    Parameters:
        dist_mat (np.ndarray): Distance matrix of shape (n, n).
        familiar_rad (float): Threshold distance for familiarity.
        n (int): Number of time points.
        window_start (int, optional): Starting offset for the sliding window relative to t.
        window_end (int, optional): Ending offset for the sliding window relative to t (exclusive).
    Returns:
        list: Proportion of familiar distances for each time point.
    """
    percent_familiar = []
    if not n:
        n = dist_mat.shape[0]
    for t in range(n):
        start_idx = 0 if window_start is None else max(0, t - window_start) # Default to all time points up to t
        end_idx = t if window_end is None else  max(0, t - window_end) # default to gap=0, otherwise go up to t - window_end
        distances = dist_mat[t, start_idx:end_idx]
        percent_familiar.append(np.mean(distances < familiar_rad) if distances.size > 0 else 0)
    
    return np.array(percent_familiar)


def get_familiarity_timeseries(all_embeddings, consec_dist, gap, n_jobs):
    """
    For each embedding, get the "familiarity-novelty" timeseries, operationalized above.
    Timepoints are counted as familiar if they are beyond a `familiarity_radius` from the previoius (excluding a `gap`). 
    """
    all_dist_mats = compute_dist_mats(all_embeddings, n_jobs=n_jobs)

    all_ts = [ 
        compute_percent_familiar(dist_mat, 
                                 familiar_rad=consec_dist['mu'][i][gap], 
                                 window_start=None, 
                                 window_end=gap)[:,np.newaxis]
        for i,dist_mat in enumerate(all_dist_mats)
    ]
    return all_ts


if __name__ == "__main__":
    N_JOBS = 8
    MODEL_NAME = 'vit' # 'vit' or 'resnet' # respectively, these will make 768-D or 2048-D embeddings
    DOWNSAMPLED_FR = 3
    OUTPUT_DIR = 'outputs'
    DEVICE = 'cpu' # 'cpu' or 'cuda'

    # Load embeddings if not already loaded
    embeddings_paths = sorted(glob.glob(OUTPUT_DIR + f'/video_embeddings/*/*{MODEL_NAME}*.npy'))
    all_embeddings = [np.load(e) for e in embeddings_paths]

    # RAW AUTOCORRELATION
    _ = run_plot_acf(all_embeddings, permute=True, n_jobs=N_JOBS, plot=True, save_folder=OUTPUT_DIR)
    # n = 50000, nlags = 20000 

    # PAIRWISE DISTANCES
    consec_dist = get_consec_dists(all_embeddings, save_folder=OUTPUT_DIR, plot=True)

    # FAMILIARITY/NOVELTY AUTOCORRELATION
    for gap in [2, 8, 32, 128]:
        familiarity_ts = get_familiarity_timeseries(all_embeddings, consec_dist, gap, n_jobs=N_JOBS)
        _ = run_plot_acf(familiarity_ts,  n=None, nlags=None, permute=False, n_jobs=N_JOBS, plot=True, 
                            save_folder=OUTPUT_DIR, save_tag = f'gap-{gap}')




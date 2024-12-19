"""Functions to calculate autocorrelation measures on model embeddings"""

import os, glob
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances
import statsmodels.api as sm
from joblib import Parallel, delayed
from utils import compute_stats, pickle_load_dict, pickle_save_dict

sns.set(style='white', palette='colorblind', context='talk')
plt.rcParams['figure.dpi'] = 200
plt.rcParams['agg.path.chunksize'] = 10000


def plot_consec_dist(consec_dist, fpath=''):
    """Plotting helper for below"""

    f, ax = plt.subplots(1, 1, figsize=(7, 4))
    f.suptitle('average pairwise Euclidean distance by gap')

    if len(consec_dist['mu']) > 1:
        cmap = plt.cm.viridis  # Choose your preferred colormap
        colors = cmap(np.linspace(0, 1, len(consec_dist['mu'])))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(consec_dist['mu'])))
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('Session date')
        cbar.set_ticks([0, len(consec_dist['mu'])])  
        cbar.ax.set_yticklabels(['early', 'late']) 
    else:
        colors = ['gray']
    
    for i, (norms_mu, norms_se) in enumerate(zip(consec_dist['mu'], consec_dist['se'])):
        mu = list(norms_mu.values())
        se = list(norms_se.values())
        ax.plot(mu, label=f'video {i}', color=colors[i])
        ax.fill_between(
            range(len(mu)), 
            np.array(mu) - np.array(se), 
            np.array(mu) + np.array(se), 
            color='gray', alpha=0.2, zorder=-99, label=f'Uncertainty' if i == 0 else None
        )

    ax.set_xticks(range(len(consec_dist['mu'][0])))  # Adjust based on the length of `mu`
    ax.set_xticklabels(consec_dist['mu'][0].keys())  # Replace `gaps` with your labels if available
    ax.set_xlabel('Video frame gap')
    ax.set_ylabel('Embedding distance')
    # ax.legend()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    sns.despine()
    f.tight_layout()

    if fpath:
        plt.savefig(fpath+'.png')

    plt.show()


def get_consec_dists(all_embeddings, plot=True, save_folder=None, save_tag=''):
    """ 
    gets consecutive distances between pairs of points
    works for either one array of embeddings or a list of them
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

    fpath = None
    if save_folder:
        out_dir = os.path.join(save_folder, 'consec_dist')
        os.makedirs(out_dir, exist_ok=True)
        fpath = os.path.join(out_dir, 'consec_dist')
        if save_tag: fpath += '-' + save_tag
        pickle_save_dict(consec_dist, fpath+'.pkl')

    if plot:
        plot_consec_dist(consec_dist, fpath)

    return consec_dist


def compute_acf_across_dims(embeddings, nlags, perm=None, missing='conservative'):
    if len(embeddings.shape) < 2:
        embeddings = embeddings[:,np.newaxis]
    dim = embeddings.shape[1]
    if perm is not None:
        embeddings = embeddings[perm]
    acf_ndim_perm = np.array([sm.tsa.acf(embeddings[:, d], nlags=nlags, missing=missing) for d in range(dim)])
    return acf_ndim_perm.mean(axis=0)


def plot_acf(acfs_all, acfs_perm_mu_se_all=[], plot_ylims=(None, None), 
             plot_timepoints=['1s','10s','1m','10m','1h','10h','1d','10d'], 
             fpath=None, raw_bool=True, save_tag=''):
    """Plotting helper for below"""
    if 'raw' in fpath:
        raw_bool=True
    if 'fn' in fpath:
        raw_bool=False

    plot_title = 'Autocorrelation of embeddings, (avg across units)' if raw_bool else 'Autocorrelation of familiarity timeseries' 
    plot_title += f' ({save_tag})'
    # get the timepoints to label
    # we could've used the timepoints directly to get this, but honestly this was just easier and it doesn't matter if its a couple frames off
    plot_timepoints_in_seconds = [int(t[:-1]) if 's' in t else int(t[:-1])*60 if 'm' in t else int(t[:-1])*3600 if 'h' in t else int(t[:-1])*86400 if 'd' in t else 0 for t in plot_timepoints]
    frame_rate = 3
    max_timepoint = (len(acfs_all[0]) - 1) / frame_rate # (seconds) 
    lag_indices_to_label = [int(t * frame_rate) for t in plot_timepoints_in_seconds if t < max_timepoint]

    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    f.suptitle(plot_title)

    if len(acfs_all) > 1:
        cmap = plt.cm.viridis  
        colors = cmap(np.linspace(0, 1, len(acfs_all)))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(acfs_all)))
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('Session date')
        cbar.set_ticks([0, len(acfs_all)])  
        cbar.ax.set_yticklabels(['early', 'late']) 
    else:
        colors = ['gray']

    for i,acf in enumerate(acfs_all):
        # Plot the ACF for the current array
        ax.plot(acf[1:], color=colors[i]) #, label=f'ACF {i+1}') 
        if len(acfs_perm_mu_se_all) > 0:
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
    ax.set_ylim(bottom=plot_ylims[0], top=plot_ylims[1])
    ax.set_xticks(lag_indices_to_label)
    ax.set_xticklabels(plot_timepoints[:len(lag_indices_to_label)])
    if len(acfs_perm_mu_se_all) > 0: ax.legend()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    sns.despine()
    f.tight_layout()

    if fpath:
        plt.savefig(fpath+'.png')
    
    plt.show()


# main function
def run_plot_acf(all_embeddings,  n=None, nlags=None, permute_n_iter=0, n_jobs=1, plot=True, 
                 plot_timepoints=['1s','10s','1m','10m','1h','10h','1d','10d'], plot_ylims=(None,None),
                 save_folder=None, save_tag=''):
    """
    Calculates autocorrelation of data and plots!
    Inputs:
        - all_embeddings: list of np.array embeddings of length n_videos and each element shape (timepoints, dims) 
                For the familiarity-novelty timeseries, dims=1 -- still works!
        - n: number of timepoints to consider
        - nlags: lags of the autocorrelation function
        - permute_n_iter: compute a permuted null by shuffling and recomputing ACF 
        - n_jobs: number of jobs to parallelize across
        - bool to plot
        - plot_timepoints are desired timepoints to show on the plot 
        - save_folder: if provided, save outputs and plots there
    This works for both the raw data and familiarity timeseries
    """
    if not type(all_embeddings) == list:
        all_embeddings = [all_embeddings] # make this compatible with a single continuous embeddings file or list of files
    
    raw_bool = all_embeddings[0].shape[1] > 1
    if raw_bool:
        print('Computing autocorrelation of raw embeddings...')
    else:
        print('Computing autocorrelation of familiarity timeseries...')
        
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

        if permute_n_iter > 0:
            # permute the embedding units and compute autocrrelation 
            acf_perm = np.array(
                Parallel(n_jobs=n_jobs)(
                    delayed(compute_acf_across_dims)(embeddings, nlags, perm=np.random.permutation(n)) for _ in range(permute_n_iter)
                )
            )
            acf_perm_mu, acf_perm_se = compute_stats(acf_perm, axis=0)
            # w = 10
            # acf_perm_mu_smoothed = moving_average(acf_perm_mu, w)
            acfs_perm_mu_se_all.append((acf_perm_mu, acf_perm_se))

    fpath = None
    if save_folder:
        save_folder_addition = 'acf_raw' if raw_bool else 'acf_fn'
        acf_out_dir = os.path.join(save_folder, save_folder_addition)
        os.makedirs(acf_out_dir, exist_ok=True)
        if len(acfs_all) > 1:
            fpath = os.path.join(acf_out_dir, 'acfs_all')
        else:
            fpath = os.path.join(acf_out_dir, 'acfs_concat')
        if save_tag: fpath += '-' + save_tag
        pickle_save_dict({'acfs_all': acfs_all, 'acfs_perm_mu_se_all': acfs_perm_mu_se_all}, fpath+'.pkl')
        
    if plot:
        plot_acf(acfs_all, acfs_perm_mu_se_all, plot_ylims, plot_timepoints, fpath, raw_bool=raw_bool, save_tag=save_tag)

    return acfs_all, acfs_perm_mu_se_all


# def compute_dist_mats(all_embeddings,  n_jobs):
#     """Euclidean distance of each pair of timepoints"""
#     if n_jobs > 1 and len(all_embeddings) > 1:
#         return Parallel(n_jobs=n_jobs)(
#             delayed(lambda e: distance.squareform(distance.pdist(e, 'euclidean')))(e) for e in all_embeddings
#         )
#     else:
#         return [distance.squareform(distance.pdist(e, 'euclidean')) for e in all_embeddings]
    

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
    if not isinstance(all_embeddings, list):
        all_embeddings = [all_embeddings]

    # this is very time consuming
    all_dist_mats = [pairwise_distances(e, metric='nan_euclidean', n_jobs=n_jobs) for e in all_embeddings] # compute_dist_mats(all_embeddings, n_jobs=n_jobs)
    # ^^ looking for a better way to do this but don't have one yet
    
    all_ts = [ 
        compute_percent_familiar(dist_mat, 
                                 familiar_rad=consec_dist['mu'][i][gap], 
                                 window_start=None, 
                                 window_end=gap)[:,np.newaxis]
        for i,dist_mat in enumerate(all_dist_mats)
    ]
    return all_ts


if __name__ == "__main__":
    INPUT_DIR = 'videos'
    OUTPUT_DIR = 'outputs'
    CONCATENATE_ALL = False

    DOWNSAMPLED_FR = 3
    MODEL_NAME = 'vit' # 'vit' or 'resnet' # respectively, these will make 768-D or 2048-D embeddings
    DEVICE = 'cpu' # 'cpu' or 'cuda'
    N_JOBS = 8
    PERMUTE_N_ITER = 10

    # Load embeddings if not already loaded
    if CONCATENATE_ALL:
        all_embeddings = pickle_load_dict(OUTPUT_DIR + f'/video_embeddings/all_embeddings-{MODEL_NAME}.pkl')['embeddings']
    else:
        all_embeddings = [pickle_load_dict(e)['embeddings'] for e in sorted(glob.glob(OUTPUT_DIR + f'/video_embeddings/[!all_embeddings]*{MODEL_NAME}*.pkl'))]

    # RAW AUTOCORRELATION
    _ = run_plot_acf(all_embeddings, permute_n_iter=PERMUTE_N_ITER, n_jobs=N_JOBS, plot=True, 
                     plot_ylims=(1e-7,None), save_folder=OUTPUT_DIR, save_tag=MODEL_NAME)


    # PAIRWISE DISTANCES
    consec_dist = get_consec_dists(all_embeddings, plot=True, save_folder=OUTPUT_DIR, save_tag=MODEL_NAME)

    # FAMILIARITY/NOVELTY AUTOCORRELATION
    for gap in [2, 8, 32, 128]:
        familiarity_ts = get_familiarity_timeseries(all_embeddings, consec_dist, gap, n_jobs=N_JOBS)
        _ = run_plot_acf(familiarity_ts,  n=None, nlags=None, permute_n_iter=PERMUTE_N_ITER, n_jobs=N_JOBS, 
                        plot=True, save_folder=OUTPUT_DIR, save_tag = f'{MODEL_NAME}-gap{gap}')




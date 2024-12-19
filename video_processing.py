"""functions for reading videos and embedding with transformer model"""

import os, glob, cv2
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import AutoImageProcessor, ResNetModel, ViTImageProcessor, ViTModel
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from utils import pickle_save_dict, pickle_load_dict, datetime_to_string, string_to_datetime


def get_cropping_dims(cap, square_len=420, vertical_offset=30):
    """
    Calculate cropping dimensions
        - cap is from cv2.VideoCapture(path)
        - square_len is the side length of the square crop
        - vertical_offset is the amount to lower below center
    """
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    center_x, center_y = frame_width // 2, frame_height // 2
    adjusted_center_y = center_y + vertical_offset
    
    start_x =  max(center_x - square_len // 2, 0)
    start_y = max(adjusted_center_y - square_len // 2, 0)
    end_x = min(start_x + square_len, frame_width)
    end_y = min(start_y + square_len, frame_height)

    return {'start_x': start_x, 'start_y': start_y, 'end_x': end_x, 'end_y': end_y}
    

def preprocess_frame(frame, crop_dims):
    """Crop a single frame to a central square: corrects for some of the fisheye distortion and crops out the timestamp"""
    frame = frame[::-1, ::-1, ::-1] # the frames come flipped (including color channel), so flip them back
    frame = frame[crop_dims['start_y']:crop_dims['end_y'], crop_dims['start_x']:crop_dims['end_x'], :] # crop
    return frame


def load_model(model_name, device='cpu'):
    """Loads either resnet or ViT model and image processor from Hugging Face."""
    if model_name.lower() == 'vit':
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    elif model_name.lower() == 'resnet':
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        model = ResNetModel.from_pretrained("microsoft/resnet-50")

    model.eval().to(device)
    return model, processor

############################ USE THESE 3 FUNCTIONS IF RUNNING ON CPU #############################
def model_transform_single_frame(frame, model, processor, model_name):
    with torch.no_grad():
        inputs = processor(images=frame, return_tensors="pt")
        hidden = model(**inputs).last_hidden_state
    if model_name.lower() == 'vit':
        # ***FOR THE ViT MODEL***
        # the last hidden state is of shape (num_batches=1, num_patches incl class token, num_units=768)
        # we just want the 'class token' representation, not image patches -- best proxy for a single representation
        output = hidden[:, 0, :]  # Shape: (1, 768)
    elif model_name.lower() == 'resnet':
        # ***FOR THE RESNET MODEL***
        # the last hidden state is of shape (num_batches, num_units=2048, 7,7 )
        # Apply Global Average Pooling: Reduces (1, 2048, 7, 7) -> (1, 2048)
        output = F.adaptive_avg_pool2d(hidden, (1, 1)).squeeze(-1).squeeze(-1)
    return output.squeeze(0).detach().cpu().numpy()


def load_and_embed_frames(target_times, video_path, crop_dims, preprocess, model_name):
    """
    Take a set of target times, open the video, crop/preprocess them and transform with model if necessary.
    If model_name is None -> returns the frames themselves
    Inputs:
        - target_times: list or array of target timepoints of the frames to load
        - video_path: path of the video to load
        - crop_dims: passed to preprocess_frame to crop properly
        - preprocess: bool to preprocess or not
        - model_name: 'vit' or 'resnet' name of the model to use. If none -> returns the frames themselves 
    Output: list of frames with preprocessing and/or model embedding performed to them
    """
    if model_name: # ['vit','resnet']
        model, processor = load_model(model_name)
    cap = cv2.VideoCapture(video_path)

    frames = []
    for target_time in target_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, target_time * 1000)
        ret, frame = cap.read()
        if not ret:  # End of video
            break
        if preprocess:
            frame = preprocess_frame(frame, crop_dims)
        if model_name:
            frame = model_transform_single_frame(frame, model, processor, model_name)
        frames.append(frame)

    cap.release()
    return frames


def load_and_embed_frames_parallel(target_times, video_path, crop_dims, preprocess, model_name, n_jobs):
    """Wrapper for above, with parallelization"""
    if n_jobs == 1:
        return load_and_embed_frames(target_times, video_path, crop_dims, preprocess, model_name)
    else:
        target_chunks = np.array_split(target_times, n_jobs)
        frames = Parallel(n_jobs=n_jobs)(
            delayed(load_and_embed_frames)(chunk_times, video_path, crop_dims, preprocess, model_name) # no model_name 
            for chunk_times in target_chunks
        )
        return np.array([item for chunk in frames for item in chunk if item is not None])
        

############################ IF WE CAN USE GPU #############################
def embed_frames_in_batches(frames, model_name, batch_size, device='cuda'):
    """If video frames are loaded separately, we can pass them to the model (separately) on GPU"""
    model, processor = load_model(model_name, device)
    embeddings = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device, non_blocking=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        if model_name.lower() == 'vit':
            batch_embeddings = outputs.last_hidden_state[:, 0, :]  # Class token embeddings
        elif model_name.lower() == 'resnet':
            batch_embeddings = F.adaptive_avg_pool2d(outputs.last_hidden_state, (1, 1)).squeeze(-1).squeeze(-1) # apply pooling across 7x7 channels

        embeddings.append(batch_embeddings.cpu())  # Bring embeddings back to CPU for final storage

    return torch.cat(embeddings).cpu().numpy()
###########################################################################


# main function
def read_embed_video(video_folder_path, n_frames=None, downsampled_frame_rate=None,
                     preprocess=True, model_name=None, save_folder=None, n_jobs=1, device='cpu'):
    """
    Reads a video, preprocesses frames, and optionally embeds them using a model.
    Dynamically adapts for CPU parallelization or GPU batch processing.

    Parameters:
        - video_folder_path: string, path to either single video to be embedded or folder of videos
        - video_paths: list of videos to be concatenated. also works with just a single string.
        - n_frames (int, optional): Number of frames to load from the video.
        - downsampled_frame_rate (float, optional): Frame rate (Hz) for downsampling.
        - preprocess (bool): Whether to preprocess frames.
        - model_name: if provided, transform frame to last hidden state of this model ['vit', 'resnet']
        - save_folder (str, optional): path to save output.
        - n_jobs (int, optional): Number of parallel jobs to run (CPU only).
        - device (str): 'cpu' or 'cuda' to specify computation device.

    Returns:
        np.array of processed frames or embeddings.
    """
    # check input video path string
    if os.path.isdir(video_folder_path): # directory
        video_paths = [os.path.join(video_folder_path, f) for f in sorted(os.listdir(video_folder_path)) if f.endswith('.mp4')]
    elif video_folder_path.endswith('.mp4'): # single video file
        video_paths = [video_folder_path]
        video_folder_path = video_folder_path[:-4] # take off .mp4 tag
    else:
        print('Unrecognized input string.')
        return

    ALL_OUTPUT_FRAMES = []
    for video_path in tqdm(video_paths):

        # first, open the video to get the video metadata we need (to calculate the crop, and the downsampled target times)
        cap = cv2.VideoCapture(video_path)
        original_frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / original_frame_rate

        frame_rate = downsampled_frame_rate if downsampled_frame_rate else original_frame_rate
        # sample target times evenly from the total video duration, we will then choose the next frame from each target time
        target_times = np.arange(0, video_duration, 1 / frame_rate)

        if n_frames:
            # if only returning n_frames, cut off the target times there
            target_times = target_times[:n_frames]

        # get the parameters necessary to crop each from, passed to preprocess_frame()
        crop_dims = get_cropping_dims(cap)
        cap.release()

        # next, if a model_name is provided, embed the frames of the video file (in parallel)
        if model_name and device == 'cuda':
            # Use batch processing on GPU. So, we'll load the videos first separately with no embedding (this is parallelized on CPU)
            frames_preprocess_only = load_and_embed_frames_parallel(target_times, video_path, crop_dims, preprocess, model_name=None, n_jobs=n_jobs) # no model_name 
            # now, pass to the model on GPU
            frames_out = embed_frames_in_batches(frames_preprocess_only, model_name, batch_size=32, device=device)

        else: # model_name is None and/or running on 'cpu'
            # if model_name not provided, this function will handle it
            frames_out = load_and_embed_frames_parallel(target_times, video_path, crop_dims, preprocess, model_name, n_jobs)

        # take all the frames from this video file and append to our full-video (directory) frames
        # each element (frames_out) is a numpy array
        ALL_OUTPUT_FRAMES.append(frames_out)
    
    ALL_OUTPUT_FRAMES = np.concatenate(ALL_OUTPUT_FRAMES, axis=0) # concatenate along time axis

    #### get timepoints of each frame
    filename_only = video_folder_path[video_folder_path.rfind('/')+1:]
    start_idx = len(video_folder_path) - len(filename_only) + filename_only.find('_') # right after the first underscore (after last /) comes the date YYYYMMDD_HHMM
    # THE VIDEOS DO NOT HAVE TIMES, ONLY DATES -- we will mark them as just starting at midnight
    video_start_time = string_to_datetime(video_folder_path[start_idx+1:start_idx+9], pattern="%Y%m%d")

    datetime_array = [video_start_time + timedelta(seconds=i / frame_rate) for i in range(len(ALL_OUTPUT_FRAMES))]
    timestamp_array = np.array([datetime_to_string(t, truncate_digits=4) for t in datetime_array])
    ## ^^ this is now proper timestamps of each frame

    # Save output if save_folder is provided
    if save_folder:
        output_dict = {'embeddings': ALL_OUTPUT_FRAMES, 'timestamps': timestamp_array}
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, video_folder_path.split('/')[-1]) + f'-{model_name}.pkl'
        pickle_save_dict(output_dict, save_path)

    return ALL_OUTPUT_FRAMES, timestamp_array


def plot_frames(images, titles=None, crop_dims=None):
    if not isinstance(images, list):
        images = [images]
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1: 
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(images[i])

        # Draw rectangle if crop_dims is provided
        if i % 2 == 0 and crop_dims is not None:
            rect = patches.Rectangle(
                (crop_dims['start_x'], crop_dims['start_y']),  # Bottom-left corner
                crop_dims['end_x'] - crop_dims['start_x'],    # Width
                crop_dims['end_y'] - crop_dims['start_y'],    # Height
                linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)

        if titles is not None:
            ax.set_title(titles[i])
        ax.axis('off')  # Hide axes for a cleaner look
    plt.tight_layout()
    plt.show()


def concatenate_embeddings_timestamps(embeddings, timestamps, downsampled_frame_rate=3, save_path=None):
    """
    Our goal is to concatenate embeddings by their timestamps, so that we have one continuous (true time)
    array of embeddings, with NaNs filled in when we have no observations.
    Inputs:
        - embeddings: list or array of embeddings (timepointsxdims)
        - timestamps: list or array of timestamps of each frame (YYYYMMDD_HHMM_SSS.S)
        ^^ these two both match the format of the pickle dicts saved by read_embed_video
    Output:
        - embeddings: np.array (timepointsxdims)
        - timestamps: np.array of timestamps of each frame
    """
    # our goal is to create a continuous timeline from the first frame to the last, and fill in embeddings where we have them
    
    min_time = string_to_datetime(timestamps[0][0]) # "%Y%m%d_%H%M_%S.%f"
    max_time = string_to_datetime(timestamps[-1][-1])

    evenly_spaced_timestamps = [min_time + timedelta(seconds=i / downsampled_frame_rate) 
                                for i in range(int((max_time - min_time).total_seconds() * downsampled_frame_rate) + 1)]
    ground_truth_timestamps = np.array([datetime_to_string(t, truncate_digits=4) for t in evenly_spaced_timestamps]) # convert back to strings
    # this is now our 'ground truth' timeline

    # now get a new embedding list with NaNs where we don't have a matching timestamp
    timestamp_to_embedding_map = dict(zip(np.concatenate(timestamps), np.concatenate(embeddings))) # lookup
    ground_truth_embeddings = np.array([
        timestamp_to_embedding_map.get(t, np.full((768,), np.nan)) for t in ground_truth_timestamps
    ])

    if save_path: 
        pickle_save_dict({'embeddings': ground_truth_embeddings, 'timestamps': ground_truth_timestamps}, save_path)

    return ground_truth_embeddings, ground_truth_timestamps



if __name__ == "__main__":
    INPUT_DIR = 'videos'
    OUTPUT_DIR = 'outputs'

    DOWNSAMPLED_FR = 3
    MODEL_NAME = 'vit' # 'vit' or 'resnet' # respectively, these will make 768-D or 2048-D embeddings
    DEVICE = 'cpu' # 'cpu' or 'cuda'
    N_JOBS = -1
    CONCATENATE_ALL = False

    save_folder = os.path.join(OUTPUT_DIR, 'video_embeddings')

    for i,subfolder in enumerate(os.listdir(INPUT_DIR)):
        subfolder_path = os.path.join(INPUT_DIR, subfolder)

        # each subfolder corresponds to one "video" with multiple video files within to be concatenated
        if os.path.isdir(subfolder_path):

            if os.path.exists(f'{save_folder}/{subfolder}-{MODEL_NAME}.pkl'):
                print(f'File already exists for video {subfolder}, skipping...')
            
            else: 
                print(f'Beginning video {i+1} out of {len(os.listdir(INPUT_DIR))}')
                embeddings = read_embed_video(subfolder_path, 
                                            n_frames=None, 
                                            downsampled_frame_rate=DOWNSAMPLED_FR, 
                                            preprocess=True, 
                                            model_name=MODEL_NAME, 
                                            save_folder = save_folder, 
                                            n_jobs=N_JOBS, 
                                            evice=DEVICE)
                print(f'Saved results to {save_folder}, shape: {embeddings.shape}')

    if CONCATENATE_ALL:
        all_embeddings_path = f'{OUTPUT_DIR}/video_embeddings/all_embeddings-{MODEL_NAME}.pkl'
        if os.path.exists(all_embeddings_path):
            print('Concatenated file already exists, loading...')
            all_dict = pickle_load_dict(all_embeddings_path)
            all_embeddings = all_dict['embeddings']
            all_timestamps = all_dict['tiimestamps']
        else:
            # concatenate across all embeddings into one giant thing
            embeddings_paths = sorted(glob.glob(OUTPUT_DIR + f'/video_embeddings/*{MODEL_NAME}*.pkl'))
            all_dicts = [pickle_load_dict(e) for e in embeddings_paths]
            all_embeddings, all_timestamps = concatenate_embeddings_timestamps([d['embeddings'] for d in all_dicts], 
                                                                [d['timestamps'] for d in all_dicts],
                                                                downsampled_frame_rate=DOWNSAMPLED_FR,
                                                                save_path = f'{OUTPUT_DIR}/video_embeddings/all_embeddings-{MODEL_NAME}.pkl')
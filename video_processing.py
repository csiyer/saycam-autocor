"""functions for reading videos and embedding with transformer model"""

import os, glob, cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, ResNetModel, ViTImageProcessor, ViTModel
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed


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

    return torch.cat(embeddings).numpy()
###########################################################################


# main function
def read_embed_video(video_path, n_frames=None, downsampled_frame_rate=None,
                     preprocess=True, model_name=None, save_folder=None, n_jobs=1, device='cpu'):
    """
    Reads a video, preprocesses frames, and optionally embeds them using a model.
    Dynamically adapts for CPU parallelization or GPU batch processing.

    Parameters:
        - video_path (string)
        - n_frames (int, optional): Number of frames to load from the video.
        - downsampled_frame_rate (float, optional): Frame rate (Hz) for downsampling.
        - preprocess (bool): Whether to preprocess frames.
        - model_name: if provided, transform frame to last hidden state of this model ['vit', 'resnet']
        - save_folder (str, optional): Folder to save output.
        - n_jobs (int, optional): Number of parallel jobs to run (CPU only).
        - device (str): 'cpu' or 'cuda' to specify computation device.

    Returns:
        np.array of processed frames or embeddings.
    """
    cap = cv2.VideoCapture(video_path)
    original_frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / original_frame_rate

    if downsampled_frame_rate:
        target_times = np.arange(0, video_duration, 1 / downsampled_frame_rate)
    else:
        target_times = np.arange(total_frames) / original_frame_rate

    if n_frames:
        target_times = target_times[:n_frames]

    crop_dims = get_cropping_dims(cap)
    cap.release()

    if model_name and device == 'cuda':
        # Use batch processing on GPU. So, we'll load the videos first separately with no embedding
        frames_preprocess_only = load_and_embed_frames_parallel(target_times, video_path, crop_dims, preprocess, model_name=None, n_jobs=n_jobs) # no model_name 
        # now, pass to the model on GPU
        frames_out = embed_frames_in_batches(frames_preprocess_only, model_name, batch_size=32, device=device)

    else: # model_name is None and/or running on 'cpu'
        frames_out = load_and_embed_frames_parallel(target_times, video_path, crop_dims, preprocess, model_name, n_jobs) 

    # Save output if save_folder is provided
    if save_folder:
        save_path = os.path.join(save_folder, video_path[:-4])
        if model_name:
            save_path += '_' + model_name
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, frames_out)

    return frames_out


def plot_frames(images, titles=None):
    if not isinstance(images, list):
        images = [images]
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1: 
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        if titles is not None:
            ax.set_title(titles[i])
        ax.axis('off')  # Hide axes for a cleaner look
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    N_JOBS = 8
    MODEL_NAME = 'vit' # 'vit' or 'resnet' # respectively, these will make 768-D or 2048-D embeddings
    DOWNSAMPLED_FR = 3
    OUTPUT_DIR = 'outputs'
    DEVICE = 'cpu' # 'cpu' or 'cuda'

    # compute and save model embeddings of all frames in the test videos
    video_paths = sorted(glob.glob('33*/*.mp4'))

    save_folder = os.path.join(OUTPUT_DIR, 'video_embeddings')
    
    for i,video in enumerate(video_paths):
        print(f'Beginning video {i+1} out of {len(video_paths)}')
        embeddings = read_embed_video(video, n_frames=None, downsampled_frame_rate=DOWNSAMPLED_FR, preprocess=True, 
                                    model_name=MODEL_NAME, save_folder=save_folder, n_jobs=N_JOBS, device=DEVICE)
        
        if save_folder:
            print(f'Saving results to {save_folder}, shape: {embeddings.shape}')
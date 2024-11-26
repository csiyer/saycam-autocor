"""functions for reading videos and embedding with transformer model"""

import os, glob, cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


def load_model(model_name):
    """Loads either resnet or ViT model and image processor from Hugging Face."""
    if model_name.lower() == 'vit':
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    elif model_name.lower() == 'resnet':
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        model = ResNetModel.from_pretrained("microsoft/resnet-50")

    model.eval()
    return model, processor


def embed_frame_model(frame, model, processor, model_name):
    with torch.no_grad():
        inputs = processor(images=frame, return_tensors="pt")
        if model_name.lower() == 'vit':
            # ***FOR THE ViT MODEL***
            # the last hidden state is of shape (num_batches=1, num_patches incl class token, num_units=768)
            # we just want the 'class token' representation, not image patches -- best proxy for a single representation
            output = model(**inputs).last_hidden_state[:, 0, :]  # Shape: (1, 768)
        
        elif model_name.lower() == 'resnet':
            # ***FOR THE RESNET MODEL***
            # the last hidden state is of shape (num_batches, num_units=2048, 7,7 )
            # Apply Global Average Pooling: Reduces (1, 2048, 7, 7) -> (1, 2048)
            feature_map = model(**inputs).last_hidden_state
            output = F.adaptive_avg_pool2d(feature_map, (1, 1)).squeeze(-1).squeeze(-1)
            
    return output.squeeze(0).detach().cpu().numpy()


def process_frame_batch(target_times, video_path, crop_dims, preprocess, model_name):
    """HELPER FOR BELOW - Processes a batch of frames. Loads model for this chunk separately."""
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
            frame = embed_frame_model(frame, model, processor, model_name)
        frames.append(frame)

    cap.release()
    return frames


def read_embed_video(video_path, n_frames=None, downsampled_frame_rate=None,
                            preprocess=True, model_name=None, save_folder=None, n_jobs=4):
    """
    Reads a video, preprocesses frames, and optionally embeds them using the ViT model.
    Parallelized with joblib.

    Parameters:
        - video_path (string)
        - n_frames (int, optional): Number of frames to load from the video.
        - downsampled_frame_rate (float, optional): Frame rate (Hz) for downsampling.
        - preprocess (bool): Whether to preprocess frames.
        - model_name: if provided, transform frame to last hidden state of this model ['vit', 'resnet']
        - save_folder (str, optional): Folder to save output.
        - n_jobs (int, optional): Number of parallel jobs to run.

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

    # Split target times into chunks for each worker
    target_chunks = np.array_split(target_times, n_jobs)

    # Parallel processing using joblib with initializer for model/processor
    processed_frames = Parallel(n_jobs=n_jobs)(
        delayed(process_frame_batch)(chunk, video_path, crop_dims, preprocess, model_name)
        for chunk in target_chunks
    )
    frames_array = np.array([item for chunk in processed_frames for item in chunk if item is not None])

    # Save output if save_folder is provided
    if save_folder:
        save_path = os.path.join(save_folder, video_path[:-4])
        if model_name:
            save_path += '_' + model_name
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, frames_array)

    return frames_array


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
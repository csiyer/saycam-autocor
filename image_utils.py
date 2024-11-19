import numpy as np
import cv2 
import matplotlib.pyplot as plt

# these are my own

def get_video_info(cap):
    """Returns video information"""
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # leave this as a float
    frame_length = frame_count // frame_rate
    return frame_count, frame_width, frame_height, frame_rate, frame_length


def crop_frame(frame, square_len = 420, vertical_offset = 30):
    """
    This function takes in a frame and crops it. It also fixes the flipped orientation. It also converts BRG format to RGB.
    This has two purposes: crop out the timestamp, and just to focus on the center of the frame.
    Inputs:
        - frame to crop (originally, the images are height 480 and width 640)
        - square_len: the side length (px) of a centered square to crop
        - offset: the distance (px) to move the square from centered
    Outputs: the cropped frame
    """
    # STEP 1: FLIP THE IMAGE (they come flipped, I don't know why)
    frame = frame[::-1, ::-1, :]

    # STEP 2: get the coordinates boudning the square
    frame_height, frame_width = frame.shape[:2]
    center_x, center_y = frame_width // 2, frame_height // 2
    adjusted_center_y = center_y + vertical_offset
    start_x = max(center_x - square_len // 2, 0) 
    start_y = max(adjusted_center_y - square_len // 2, 0)
    end_x = min(start_x + square_len, frame_width)
    end_y = min(start_y + square_len, frame_height)
    
    # STEP 3: crop
    cropped_frame = frame[start_y:end_y, start_x:end_x, :]

    # STEP 4: flip BGR to RGB
    cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB) 
    return cropped_frame


def undistort_fisheye_frame(frame, camera_matrix=None, dist_coeffs=None):
    """NOT CURRENTLY USED - DELETE!
    Uses built-in CV2 functionality to undistort the frame
    """
    if camera_matrix is None or dist_coeffs is None: # Default calibration values (assumed general fisheye characteristics)
        h, w = frame.shape[:2]
        focal_length = max(w, h) / 2
        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ])
        dist_coeffs = np.array([-0.3, 0.1, 0, 0])  # Approximate fisheye distortion coefficients
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, np.eye(3), camera_matrix, frame.shape[:2][::-1], cv2.CV_16SC2
    )
    undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
    return undistorted_frame


def preprocess_frame(frame, distortion_correction=True, crop=True):
    """NOT CURRENTLY BEING USED - DELETE
    Preprocesses an individual frame, with the following steps:
        1) changes default BGR to RGB
        2) undistorts the fisheye with function above
        3) crops off the times
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # RGB instead of default
    if distortion_correction:
        frame = undistort_fisheye_frame(frame)
    if crop:
        frame = crop_frame(frame)
    return frame


def load_video(video_path, n_frames=None, downsampled_frame_rate=None, crop=True):
    """
    Loads a video to np.array.
    - n_frames specifies how many frames to load (if not supplied, loads all target frames).
    - downsampled_frame_rate specifies how many frames per second (Hz) to sample.
    - cropping is performed by the helper function crop_frame.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    original_frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / original_frame_rate

    if downsampled_frame_rate:
        # print(f'Original framerate: {original_frame_rate:.2f} Hz, downsampling to {downsampled_frame_rate} Hz.')
        # Generate target timestamps at the downsampled frame rate
        target_times = np.arange(0, video_duration, 1 / downsampled_frame_rate)
    else:
        target_times = np.arange(total_frames) / original_frame_rate

    if n_frames:
        target_times = target_times[:n_frames]  # Limit to n_frames if specified

    for target_time in target_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, target_time * 1000) # Set the capture position to the desired timestamp (in milliseconds)
        ret, frame = cap.read()
        if not ret: # done with the video
            break
        if crop:
            frame = crop_frame(frame)
        frames.append(frame)
    
    cap.release()
    return np.array(frames)

def plot_frame(frame, title=''):
    plt.imshow(frame)
    plt.suptitle(title)
    plt.axis('off')  
    plt.show()

def plot_frames(images, titles=None):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        if titles is not None:
            ax.set_title(titles[i])
        ax.axis('off')  # Hide axes for a cleaner look
    plt.tight_layout()
    plt.show()
"""
Augmentation and Collation Module
Implements spatial/pixel augmentations (brightness, contrast, rotation, coarse dropout, flip)
and temporal padding to max 250 frames, as specified in Section VI-B(1c, 1d) of the base paper.
"""

import numpy as np

def apply_spatial_augmentations(video_tensor):
    """
    Applies spatial and photometric augmentations to a video tensor (T, C, H, W).
    
    Args:
        video_tensor: numpy array of shape (T, C, H, W) normalized to [0, 1]
    """
    T, C, H, W = video_tensor.shape
    augmented = video_tensor.copy()
    
    # 1. Random Brightness & Contrast
    # Factor ~ [0.85, 1.15]
    alpha = np.random.uniform(0.85, 1.15)  # contrast
    beta = np.random.uniform(-0.1, 0.1)     # brightness
    augmented = np.clip(augmented * alpha + beta, 0.0, 1.0)
    
    # 2. Horizontal Flip (Hand Dominance Invariance, Section VI-B 4)
    # Allows model to recognize left-handed and right-handed signers equally
    if np.random.rand() > 0.5:
        augmented = np.flip(augmented, axis=3).copy()
        
    # 3. Coarse Dropout / Cutout (Simulates occlusions)
    if np.random.rand() > 0.3:
        num_holes = np.random.randint(1, 3)
        for _ in range(num_holes):
            hole_h = np.random.randint(int(H * 0.1), int(H * 0.25))
            hole_w = np.random.randint(int(W * 0.1), int(W * 0.25))
            top = np.random.randint(0, H - hole_h)
            left = np.random.randint(0, W - hole_w)
            augmented[:, :, top:top+hole_h, left:left+hole_w] = 0.0
            
    # 4. Pixel Dropout
    if np.random.rand() > 0.5:
        drop_mask = np.random.rand(T, 1, H, W) > 0.03
        augmented = augmented * drop_mask
        
    return augmented


def collate_and_pad_video(frames_tensor, max_len=250):
    """
    Pads or truncates video sequence along temporal dimension to max_len (250 frames),
    matching Section VI-B(1d).
    
    Args:
        frames_tensor: numpy array of shape (T, C, H, W)
        max_len: maximum sequence length (default: 250)
        
    Returns:
        padded_tensor: numpy array of shape (max_len, C, H, W)
        actual_len: int, original number of frames before padding
    """
    T, C, H, W = frames_tensor.shape
    actual_len = min(T, max_len)
    
    if T >= max_len:
        return frames_tensor[:max_len], max_len
        
    padded_tensor = np.zeros((max_len, C, H, W), dtype=frames_tensor.dtype)
    padded_tensor[:T] = frames_tensor
    # Repeat the last frame for smooth dummy padding
    if T > 0:
        padded_tensor[T:] = frames_tensor[T - 1]
        
    return padded_tensor, actual_len


def pad_pose_sequence(pose_matrix, max_len=250):
    """
    Pads or truncates pose feature matrix (T, 443) to max_len (250).
    """
    T, F = pose_matrix.shape
    actual_len = min(T, max_len)
    
    if T >= max_len:
        return pose_matrix[:max_len], max_len
        
    padded = np.zeros((max_len, F), dtype=pose_matrix.dtype)
    padded[:T] = pose_matrix
    if T > 0:
        padded[T:] = pose_matrix[T - 1]
        
    return padded, actual_len

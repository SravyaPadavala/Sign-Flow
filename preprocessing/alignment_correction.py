"""
Alignment Correction Module
Implements the MediaPipe-based signer centralizing and 4:3 aspect ratio cropping
as specified in Section IV and Figure 1 of the base paper.
"""

import numpy as np

def compute_alignment_shift(left_ear_coords, right_ear_coords, frame_width, frame_height):
    """
    Computes the x-shift and y-shift required to center the signer based on ear midpoint.
    
    Args:
        left_ear_coords: (x, y) coordinates of left ear (normalized [0, 1] or pixel)
        right_ear_coords: (x, y) coordinates of right ear
        frame_width: int, width of video frame
        frame_height: int, height of video frame
        
    Returns:
        x_shift: pixel shift along x-axis
        y_shift: pixel shift along y-axis
        scale_factor: estimated scaling factor based on ear distance
    """
    # Midpoint of the line connecting left and right ear
    ear_mid_x = (left_ear_coords[0] + right_ear_coords[0]) / 2.0
    ear_mid_y = (left_ear_coords[1] + right_ear_coords[1]) / 2.0
    
    # Frame center
    center_x = frame_width / 2.0
    center_y = frame_height / 2.0
    
    # Shifts to align signer to center
    x_shift = center_x - ear_mid_x
    # Base paper aligns face in upper third of frame for optimal hand tracking
    target_y = frame_height * 0.30
    y_shift = target_y - ear_mid_y
    
    # Inter-ear distance for scale normalization
    ear_dist = np.sqrt((left_ear_coords[0] - right_ear_coords[0])**2 + 
                       (left_ear_coords[1] - right_ear_coords[1])**2)
    reference_ear_dist = frame_width * 0.15
    scale_factor = reference_ear_dist / (ear_dist + 1e-6)
    scale_factor = np.clip(scale_factor, 0.7, 1.4)
    
    return int(x_shift), int(y_shift), float(scale_factor)


def apply_alignment_correction(frame, x_shift=0, y_shift=0, scale=1.0, target_size=(112, 112)):
    """
    Shifts, crops to 4:3 aspect ratio, and resizes frame to target_size (112x112).
    
    Args:
        frame: numpy array of shape (H, W, C)
        x_shift: horizontal shift
        y_shift: vertical shift
        scale: scale factor
        target_size: tuple (target_w, target_h)
        
    Returns:
        processed_frame: numpy array of shape (target_h, target_w, C)
    """
    h, w, c = frame.shape
    
    # Simple affine translation + crop
    shifted = np.zeros_like(frame)
    src_x1 = max(0, -x_shift)
    src_x2 = min(w, w - x_shift)
    src_y1 = max(0, -y_shift)
    src_y2 = min(h, h - y_shift)
    
    dst_x1 = max(0, x_shift)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y1 = max(0, y_shift)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    
    dst_x2 = min(w, dst_x2)
    dst_y2 = min(h, dst_y2)
    copy_w = min(src_x2 - src_x1, dst_x2 - dst_x1)
    copy_h = min(src_y2 - src_y1, dst_y2 - dst_y1)
    
    if copy_w > 0 and copy_h > 0:
        shifted[dst_y1:dst_y1+copy_h, dst_x1:dst_x1+copy_w] = frame[src_y1:src_y1+copy_h, src_x1:src_x1+copy_w]
    else:
        shifted = frame.copy()
        
    # Crop to 4:3 aspect ratio centered
    crop_w = int(w * 0.85)
    crop_h = int(crop_w * 4 / 3)
    if crop_h > h:
        crop_h = h
        crop_w = int(crop_h * 3 / 4)
        
    start_x = max(0, (w - crop_w) // 2)
    start_y = max(0, (h - crop_h) // 2)
    cropped = shifted[start_y:start_y+crop_h, start_x:start_x+crop_w]
    
    # Bilinear downsampling to target size (112, 112)
    target_w, target_h = target_size
    y_indices = np.linspace(0, cropped.shape[0] - 1, target_h).astype(np.int32)
    x_indices = np.linspace(0, cropped.shape[1] - 1, target_w).astype(np.int32)
    resized = cropped[np.ix_(y_indices, x_indices)]
    
    return resized

"""
Dynamic Video Downsampling Module
Implements adaptive frame rate reduction to 6 fps (or random [5, 8] range for training)
with similarity-based frame selection as described in Section VI-B(1a, 1b).
"""

import numpy as np

def compute_frame_similarity(frame_a, frame_b):
    """
    Computes Normalized Mean Squared Error / Cosine similarity between two consecutive frames.
    Lower diff means higher similarity.
    """
    diff = np.mean(np.abs(frame_a.astype(np.float32) - frame_b.astype(np.float32)))
    return diff


def dynamic_downsample_video(frames, source_fps=30.0, target_fps=6.0, is_training=False, fps_range=(5, 8)):
    """
    Segments video into uniform temporal bins and selects representative frames.
    
    Args:
        frames: list or numpy array of frames (T, H, W, C)
        source_fps: original video fps
        target_fps: desired output fps (default 6 fps)
        is_training: if True, applies random fps selection within fps_range and random sampling
        fps_range: tuple of (min_fps, max_fps) for training augmentation
        
    Returns:
        downsampled_frames: numpy array of sampled frames (T_new, H, W, C)
        sampled_indices: list of frame indices selected
    """
    total_frames = len(frames)
    if total_frames == 0:
        return np.array([]), []
        
    if is_training:
        effective_target_fps = np.random.uniform(fps_range[0], fps_range[1])
    else:
        effective_target_fps = target_fps
        
    duration_sec = total_frames / float(source_fps)
    num_target_frames = max(1, int(round(duration_sec * effective_target_fps)))
    
    # Segment boundaries
    segment_bounds = np.linspace(0, total_frames, num_target_frames + 1).astype(int)
    sampled_indices = []
    
    for i in range(num_target_frames):
        start_idx = segment_bounds[i]
        end_idx = max(start_idx + 1, segment_bounds[i + 1])
        segment_len = end_idx - start_idx
        
        if is_training:
            # Training: Randomly select one frame within the segment
            selected_idx = np.random.randint(start_idx, end_idx)
        else:
            # Inference: Select the frame that has maximum motion informativeness (similarity to segment mean)
            if segment_len == 1:
                selected_idx = start_idx
            else:
                segment_frames = frames[start_idx:end_idx]
                segment_mean = np.mean(segment_frames, axis=0)
                diffs = [compute_frame_similarity(f, segment_mean) for f in segment_frames]
                # Pick the most representative frame closest to mean
                best_sub_idx = int(np.argmin(diffs))
                selected_idx = start_idx + best_sub_idx
                
        sampled_indices.append(selected_idx)
        
    sampled_frames = np.array([frames[idx] for idx in sampled_indices])
    return sampled_frames, sampled_indices

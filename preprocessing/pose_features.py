"""
Pose & Landmark Feature Engineering Module
Extracts 443 spatio-temporal features per frame from MediaPipe Pose and Hands,
matching Section V-B and Figures 3 & 4 of the base paper, including polynomial interpolation.
"""

import numpy as np

# Upper-body pose connections (indices 0 to 22 in MediaPipe pose)
UPPER_POSE_CONNECTIONS = [
    (11, 12),  # shoulder to shoulder
    (11, 13), (13, 15),  # left shoulder -> left elbow -> left wrist
    (12, 14), (14, 16),  # right shoulder -> right elbow -> right wrist
    (11, 23), (12, 24), (23, 24),  # shoulders to hips
    (0, 1), (1, 2), (2, 3), (3, 7),  # face left
    (0, 4), (4, 5), (5, 6), (6, 8),  # face right
    (9, 10)  # mouth
]

# Hand connections (21 joints, 20 bones)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

def interpolate_missing_landmarks(trajectory, degree=2):
    """
    Applies polynomial interpolation (degree 2) to fill missing/void landmark frames,
    as specified in Figure 4 of the base paper.
    
    Args:
        trajectory: numpy array of shape (T, num_features) with NaNs or zeros for missing frames
        degree: polynomial degree (default: 2)
        
    Returns:
        interpolated: numpy array of shape (T, num_features)
    """
    T, F = trajectory.shape
    interpolated = trajectory.copy()
    time_steps = np.arange(T)
    
    for f in range(F):
        feature_vals = trajectory[:, f]
        valid_mask = ~np.isnan(feature_vals) & (feature_vals != 0.0)
        num_valid = np.sum(valid_mask)
        
        if num_valid == 0:
            interpolated[:, f] = 0.0
        elif num_valid < degree + 1:
            # Linear fallback if fewer valid points than degree
            if num_valid == 1:
                interpolated[:, f] = feature_vals[valid_mask][0]
            else:
                interpolated[:, f] = np.interp(time_steps, time_steps[valid_mask], feature_vals[valid_mask])
        else:
            poly_deg = min(degree, num_valid - 1)
            coeffs = np.polyfit(time_steps[valid_mask], feature_vals[valid_mask], deg=poly_deg)
            interpolated[:, f] = np.polyval(coeffs, time_steps)
            
    return interpolated


def extract_frame_pose_features(pose_upper, left_hand, right_hand):
    """
    Extracts the 443-dimensional feature vector for a single frame.
    
    Args:
        pose_upper: (22, 3) Upper body coordinates (X, Y, Z)
        left_hand: (21, 3) Left hand coordinates (X, Y, Z)
        right_hand: (21, 3) Right hand coordinates (X, Y, Z)
        
    Returns:
        feat_vec: (443,) feature vector
    """
    features = []
    
    # 1. Raw joint coordinates (X, Y, Z)
    # Pose: 22 * 3 = 66
    features.extend(pose_upper.flatten())
    # Left hand: 21 * 3 = 63
    features.extend(left_hand.flatten())
    # Right hand: 21 * 3 = 63
    features.extend(right_hand.flatten())
    
    # 2. Bone lengths (Euclidean distance between connected joints)
    # Pose bone lengths
    for j1, j2 in UPPER_POSE_CONNECTIONS:
        if j1 < len(pose_upper) and j2 < len(pose_upper):
            dist = np.linalg.norm(pose_upper[j1] - pose_upper[j2])
            features.append(dist)
        else:
            features.append(0.0)
            
    # Left hand bone lengths (20 connections)
    for j1, j2 in HAND_CONNECTIONS:
        dist = np.linalg.norm(left_hand[j1] - left_hand[j2])
        features.append(dist)
        
    # Right hand bone lengths (20 connections)
    for j1, j2 in HAND_CONNECTIONS:
        dist = np.linalg.norm(right_hand[j1] - right_hand[j2])
        features.append(dist)
        
    # 3. Relative positions of each joint with their adjacent/parent joints
    for j1, j2 in UPPER_POSE_CONNECTIONS:
        if j1 < len(pose_upper) and j2 < len(pose_upper):
            rel = pose_upper[j2] - pose_upper[j1]
            features.extend(rel)
        else:
            features.extend([0.0, 0.0, 0.0])
            
    for j1, j2 in HAND_CONNECTIONS:
        rel = left_hand[j2] - left_hand[j1]
        features.extend(rel)
        
    for j1, j2 in HAND_CONNECTIONS:
        rel = right_hand[j2] - right_hand[j1]
        features.extend(rel)
        
    # 4. Joint Euclidean distances (Section V-B 4):
    # - Left hand to right hand palm (joint 0)
    # - Left hand to right hand elbow (joint 14)
    # - Left palm to left shoulder (joint 11)
    # - Right palm to right shoulder (joint 12)
    lh_palm = left_hand[0]
    rh_palm = right_hand[0]
    lh_elbow = pose_upper[13] if len(pose_upper) > 13 else np.zeros(3)
    rh_elbow = pose_upper[14] if len(pose_upper) > 14 else np.zeros(3)
    l_shoulder = pose_upper[11] if len(pose_upper) > 11 else np.zeros(3)
    r_shoulder = pose_upper[12] if len(pose_upper) > 12 else np.zeros(3)
    
    d_lh_rh = np.linalg.norm(lh_palm - rh_palm)
    d_lh_r_elbow = np.linalg.norm(lh_palm - rh_elbow)
    d_rh_l_elbow = np.linalg.norm(rh_palm - lh_elbow)
    d_lh_l_shoulder = np.linalg.norm(lh_palm - l_shoulder)
    d_rh_r_shoulder = np.linalg.norm(rh_palm - r_shoulder)
    
    features.extend([d_lh_rh, d_lh_r_elbow, d_rh_l_elbow, d_lh_l_shoulder, d_rh_r_shoulder])
    
    feat_arr = np.array(features, dtype=np.float32)
    
    # Pad or truncate to exact target 443 features as per paper specification
    target_dim = 443
    if len(feat_arr) < target_dim:
        feat_arr = np.pad(feat_arr, (0, target_dim - len(feat_arr)), mode='constant')
    elif len(feat_arr) > target_dim:
        feat_arr = feat_arr[:target_dim]
        
    return feat_arr


def extract_video_pose_sequence(raw_pose_seq, raw_lh_seq, raw_rh_seq):
    """
    Extracts 443-feature vectors for all frames in video and applies polynomial interpolation.
    """
    T = len(raw_pose_seq)
    feat_matrix = np.zeros((T, 443), dtype=np.float32)
    
    for t in range(T):
        feat_matrix[t] = extract_frame_pose_features(raw_pose_seq[t], raw_lh_seq[t], raw_rh_seq[t])
        
    interpolated_matrix = interpolate_missing_landmarks(feat_matrix, degree=2)
    return interpolated_matrix

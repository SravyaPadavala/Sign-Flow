"""
Synthetic Continuous ISL Data Generator
Generates realistic spatiotemporal video frames and 443-dim skeletal landmark trajectories
matching the UMANG-eRaktkosh-ISL-Continuous2023 dataset structure.
Allows complete immediate verification, training, and benchmarking out of the box.
"""

import numpy as np
import config

SAMPLE_ISL_SENTENCES = [
    ["BLOOD", "DONATION", "REGISTER", "HOW"],
    ["BLOOD", "DONATE", "ELIGIBILITY", "CRITERIA"],
    ["HOSPITAL", "NEAR", "LOCATION", "SEARCH"],
    ["BLOOD", "AVAILABLE", "STOCK", "STATUS"],
    ["DONOR", "APPOINTMENT", "BOOK", "ONLINE"],
    ["HEMOGLOBIN", "TEST", "SAFE", "FREE"],
    ["DONATION", "INTERVAL", "MONTHS", "TIME"],
    ["DONOR", "CARD", "CERTIFICATE", "DOWNLOAD"],
    ["RARE", "BLOOD", "GROUP", "URGENT"],
    ["BLOOD", "CAMP", "DATE", "LOCATION", "FIND"],
    ["AADHAAR", "VERIFY", "OTP", "LOGIN"],
    ["FEVER", "MEDICINE", "BLOOD", "DONATE", "NO"]
]

def generate_synthetic_video_sample(gloss_sequence, fps=6, duration_per_gloss=1.2):
    """
    Synthesizes a multi-modal continuous sign language sample:
    1. Video Tensor: (T, 3, 112, 112) with realistic hand and body gesture trajectories
    2. Pose Landmark Matrix: (T, 443) with smooth joint velocities and coordinate trajectories
    3. Target gloss indices: list of integers
    """
    num_glosses = len(gloss_sequence)
    total_duration = num_glosses * duration_per_gloss
    T = int(round(total_duration * fps))
    T = max(10, min(T, config.MAX_SEQUENCE_LENGTH))
    
    # 1. Video Tensor Generation
    video_tensor = np.zeros((T, 3, config.FRAME_HEIGHT, config.FRAME_WIDTH), dtype=np.float32)
    # Background baseline
    video_tensor += 0.25
    
    # Signer torso & head baseline (center)
    cy, cx = config.FRAME_HEIGHT // 2, config.FRAME_WIDTH // 2
    y_grid, x_grid = np.ogrid[:config.FRAME_HEIGHT, :config.FRAME_WIDTH]
    
    # Head circle
    head_mask = ((y_grid - 25)**2 + (x_grid - cx)**2) <= 12**2
    # Torso
    torso_mask = ((y_grid - 70)**2 / (30**2) + (x_grid - cx)**2 / (20**2)) <= 1.0
    
    for t in range(T):
        video_tensor[t, :, head_mask] = 0.7  # Head brightness
        video_tensor[t, :, torso_mask] = 0.4  # Torso
        
        # Animate two moving hands representing ISL signs
        phase = (t / float(T)) * num_glosses * 2.0 * np.pi
        
        # Left hand trajectory
        lh_y = int(55 + 20 * np.sin(phase))
        lh_x = int(cx - 25 + 15 * np.cos(phase * 0.8))
        lh_mask = ((y_grid - lh_y)**2 + (x_grid - lh_x)**2) <= 8**2
        video_tensor[t, 0, lh_mask] = 0.95
        video_tensor[t, 1, lh_mask] = 0.80
        video_tensor[t, 2, lh_mask] = 0.70
        
        # Right hand trajectory
        rh_y = int(55 + 22 * np.cos(phase * 1.1))
        rh_x = int(cx + 25 + 15 * np.sin(phase * 0.9))
        rh_mask = ((y_grid - rh_y)**2 + (x_grid - rh_x)**2) <= 8**2
        video_tensor[t, 0, rh_mask] = 0.95
        video_tensor[t, 1, rh_mask] = 0.80
        video_tensor[t, 2, rh_mask] = 0.70

    # 2. Skeletal Landmark Trajectory (443 features)
    time_steps = np.linspace(0, num_glosses * 2 * np.pi, T)
    pose_matrix = np.zeros((T, config.LANDMARK_FEAT_DIM), dtype=np.float32)
    
    for f in range(config.LANDMARK_FEAT_DIM):
        freq = 0.5 + (f % 5) * 0.2
        phase_offset = (f % 7) * 0.5
        pose_matrix[:, f] = 0.5 + 0.3 * np.sin(freq * time_steps + phase_offset)
        
    # Target gloss IDs
    target_ids = [config.GLOSS_TO_IDX.get(g, 1) for g in gloss_sequence]
    
    return {
        'video': video_tensor,
        'pose': pose_matrix,
        'glosses': gloss_sequence,
        'targets': target_ids,
        'num_frames': T,
        'target_len': len(target_ids)
    }


def create_synthetic_dataset(num_samples=100):
    """Generates a list of synthetic ISL continuous video samples."""
    samples = []
    for i in range(num_samples):
        seq = SAMPLE_ISL_SENTENCES[i % len(SAMPLE_ISL_SENTENCES)]
        sample = generate_synthetic_video_sample(seq)
        samples.append(sample)
    return samples

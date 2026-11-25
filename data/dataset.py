"""
Dataset and Batch Collation Module
Handles continuous sign language video streams, 443-dim landmark features,
and target gloss token sequences with dynamic batch collation.
"""

import numpy as np
import config
from data.synthetic_data import create_synthetic_dataset

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class ISLContinuousDataset(Dataset):
        def __init__(self, samples=None, is_train=True):
            if samples is None:
                # Fallback to high-fidelity synthetic samples for immediate execution
                self.samples = create_synthetic_dataset(num_samples=60 if is_train else 20)
            else:
                self.samples = samples
            self.is_train = is_train

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            item = self.samples[idx]
            video = item['video']  # (T, 3, 112, 112)
            pose = item['pose']    # (T, 443)
            targets = item['targets']
            glosses = item['glosses']
            
            return {
                'video': torch.from_numpy(video).float(),
                'pose': torch.from_numpy(pose).float(),
                'targets': torch.tensor(targets, dtype=torch.long),
                'glosses': glosses,
                'num_frames': len(video),
                'target_len': len(targets)
            }


    def pad_collate_fn(batch):
        """
        Collates samples into padded batch tensors:
        - Videos padded to max batch frames: (B, 3, T_max, 112, 112)
        - Poses padded to max batch frames: (B, T_max, 443)
        - Targets flattened or padded for CTCLoss
        - input_lengths: 1D tensor of original video frame counts
        - target_lengths: 1D tensor of gloss counts
        """
        batch_size = len(batch)
        input_lengths = torch.tensor([item['num_frames'] for item in batch], dtype=torch.long)
        target_lengths = torch.tensor([item['target_len'] for item in batch], dtype=torch.long)
        
        max_T = max(input_lengths).item()
        
        # Batch tensors
        videos = torch.zeros(batch_size, 3, max_T, config.FRAME_HEIGHT, config.FRAME_WIDTH)
        poses = torch.zeros(batch_size, max_T, config.LANDMARK_FEAT_DIM)
        
        all_targets = []
        all_glosses = []
        
        for i, item in enumerate(batch):
            T_i = item['num_frames']
            # item['video'] is (T, 3, H, W) -> permute to (3, T, H, W)
            v = item['video'].permute(1, 0, 2, 3)
            videos[i, :, :T_i, :, :] = v
            # If padded, repeat last frame to prevent sharp boundary artifacts
            if T_i < max_T:
                videos[i, :, T_i:, :, :] = v[:, -1:, :, :]
                
            p = item['pose']
            poses[i, :T_i, :] = p
            if T_i < max_T:
                poses[i, T_i:, :] = p[-1:, :]
                
            all_targets.append(item['targets'])
            all_glosses.append(item['glosses'])
            
        flat_targets = torch.cat(all_targets)
        
        return {
            'videos': videos,
            'poses': poses,
            'targets': flat_targets,
            'input_lengths': input_lengths,
            'target_lengths': target_lengths,
            'glosses': all_glosses
        }
else:
    class ISLContinuousDataset:
        pass
    def pad_collate_fn(batch):
        pass

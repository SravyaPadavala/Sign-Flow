"""
Baseline SignFlow Model Architecture
Exact implementation of the network proposed by M. Geetha et al. (IEEE Access 2025):
- 3D-ResNet (MC3_18) backbone for clip-level spatial feature extraction
- 8-layer standard Transformer Encoder (d_model=512, 8 heads)
- Optional Positional Encoding (paper tests both with and without positional encoding)
- Linear CTC classification head
"""

import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class PositionalEncoding(nn.Module):
        """Standard sinusoidal positional encoding used in Transformer models."""
        def __init__(self, d_model=512, max_len=500):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

        def forward(self, x):
            return x + self.pe[:, :x.size(1), :]


    class MC3_18_Backbone(nn.Module):
        """
        Mixed Convolution 3D ResNet-18 (MC3_18) representation.
        Uses 3D convolutions for early layers and 2D convolutions for later layers,
        followed by spatio-temporal pooling to output (B, T, 512).
        """
        def __init__(self, in_channels=3, feature_dim=512):
            super().__init__()
            # Initial 3D Conv block (captures short-term motion)
            self.conv1 = nn.Sequential(
                nn.Conv3d(in_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
            )
            # Mixed conv stage 1 (3D convs)
            self.stage1 = nn.Sequential(
                nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
                nn.BatchNorm3d(128),
                nn.ReLU(inplace=True)
            )
            # Mixed conv stage 2 (2D spatial convs across frames)
            self.stage2 = nn.Sequential(
                nn.Conv3d(128, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
                nn.BatchNorm3d(256),
                nn.ReLU(inplace=True)
            )
            # Mixed conv stage 3
            self.stage3 = nn.Sequential(
                nn.Conv3d(256, feature_dim, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
                nn.BatchNorm3d(feature_dim),
                nn.ReLU(inplace=True)
            )
            # Spatial 2D adaptive pooling (preserving temporal dimension T)
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        def forward(self, x):
            # Input shape: (B, C, T, H, W)
            out = self.conv1(x)
            out = self.stage1(out)
            out = self.stage2(out)
            out = self.stage3(out)
            # Pooling along spatial H and W: shape becomes (B, feature_dim, T, 1, 1)
            pooled = self.spatial_pool(out).squeeze(-1).squeeze(-1)
            # Transpose to (B, T, feature_dim) for Transformer sequence learning
            return pooled.transpose(1, 2)


    class SignFlowBaseModel(nn.Module):
        """
        Base Paper Model:
        MC3_18 3D-ResNet Backbone + 8-layer Transformer Encoder + CTC Classifier Head.
        """
        def __init__(self, vocab_size=163, d_model=512, num_heads=8, num_layers=8,
                     dropout=0.1, use_positional_encoding=False):
            super().__init__()
            self.use_positional_encoding = use_positional_encoding
            self.feature_extractor = MC3_18_Backbone(in_channels=3, feature_dim=d_model)
            
            if use_positional_encoding:
                self.pos_encoder = PositionalEncoding(d_model=d_model)
            else:
                self.pos_encoder = None
                
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=2048,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # Linear CTC Head
            self.classifier = nn.Linear(d_model, vocab_size)
            
            # Xavier weight initialization as described in paper Section VI-B(2)
            self._init_weights()

        def _init_weights(self):
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def forward(self, video):
            """
            Args:
                video: tensor of shape (B, C, T, H, W)
                
            Returns:
                logits: frame-wise gloss logits of shape (B, T, vocab_size)
                features: intermediate embeddings of shape (B, T, d_model)
            """
            # Visual feature extraction: (B, T, d_model)
            features = self.feature_extractor(video)
            
            if self.pos_encoder is not None:
                features = self.pos_encoder(features)
                
            # Sequence modeling through 8 Transformer layers
            seq_embeddings = self.transformer_encoder(features)
            
            # CTC logits
            logits = self.classifier(seq_embeddings)
            return logits, seq_embeddings
else:
    class SignFlowBaseModel:
        pass

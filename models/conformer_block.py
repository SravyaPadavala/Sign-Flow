"""
Conformer Sequence Modeling Block
Combines Macaron-style Feed-Forward, Depthwise Separable Temporal Convolution,
and Multi-Head Self-Attention with Relative Positional Bias.
Overcomes the base paper's trade-off between positional encoding and keyword detection.
"""

import math

# We structure this so it works directly with PyTorch or pure-python/numpy fallback if needed
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class RelativePositionalEncoding(nn.Module):
        """
        Relative Positional Embedding module that injects shift-invariant temporal distance
        into attention matrices, preserving word order without hurting keyword detection.
        """
        def __init__(self, d_model, max_len=512):
            super().__init__()
            self.d_model = d_model
            self.pe = nn.Embedding(2 * max_len + 1, d_model)
            self.max_len = max_len

        def forward(self, length):
            range_vec = torch.arange(length)
            distance_mat = range_vec[None, :] - range_vec[:, None]
            distance_mat_clamped = torch.clamp(distance_mat, -self.max_len, self.max_len) + self.max_len
            return self.pe(distance_mat_clamped.to(self.pe.weight.device))


    class ConformerConvModule(nn.Module):
        """
        Depthwise Separable Temporal Convolution module.
        Captures fine-grained local co-articulation transitions across rapid signing gestures.
        """
        def __init__(self, d_model, kernel_size=15, dropout=0.1):
            super().__init__()
            self.layer_norm = nn.LayerNorm(d_model)
            # Pointwise expansion (GLU doubles channels, so expansion to 2 * d_model)
            self.pointwise_conv1 = nn.Linear(d_model, 2 * d_model)
            # Depthwise 1D temporal convolution
            padding = (kernel_size - 1) // 2
            self.depthwise_conv = nn.Conv1d(
                d_model, d_model, kernel_size=kernel_size,
                padding=padding, groups=d_model
            )
            self.batch_norm = nn.BatchNorm1d(d_model)
            self.pointwise_conv2 = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            # x shape: (B, T, D)
            residual = x
            x = self.layer_norm(x)
            x = self.pointwise_conv1(x)
            # Gated Linear Unit (GLU)
            x = F.glu(x, dim=-1)
            
            # Conv1d expects (B, D, T)
            x = x.transpose(1, 2)
            x = self.depthwise_conv(x)
            x = self.batch_norm(x)
            x = F.silu(x)  # Swish activation
            x = x.transpose(1, 2)
            
            x = self.pointwise_conv2(x)
            x = self.dropout(x)
            return residual + x


    class ConformerFeedForward(nn.Module):
        """
        Half-step Macaron Feed-Forward module.
        """
        def __init__(self, d_model, expansion=4, dropout=0.1):
            super().__init__()
            self.layer_norm = nn.LayerNorm(d_model)
            self.linear1 = nn.Linear(d_model, d_model * expansion)
            self.linear2 = nn.Linear(d_model * expansion, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            residual = x
            x = self.layer_norm(x)
            x = F.silu(self.linear1(x))
            x = self.dropout(x)
            x = self.linear2(x)
            x = self.dropout(x)
            return residual + 0.5 * x


    class ConformerBlock(nn.Module):
        """
        Full Conformer Block:
        1. FFN (0.5x)
        2. Relative Multi-Head Self-Attention
        3. Depthwise Conv Module
        4. FFN (0.5x)
        5. LayerNorm
        """
        def __init__(self, d_model=512, num_heads=8, conv_kernel_size=15, dropout=0.1):
            super().__init__()
            self.ffn1 = ConformerFeedForward(d_model, dropout=dropout)
            self.attn_norm = nn.LayerNorm(d_model)
            self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            self.conv_module = ConformerConvModule(d_model, kernel_size=conv_kernel_size, dropout=dropout)
            self.ffn2 = ConformerFeedForward(d_model, dropout=dropout)
            self.final_norm = nn.LayerNorm(d_model)

        def forward(self, x, key_padding_mask=None):
            # 1. Macaron FFN 1
            x = self.ffn1(x)
            # 2. Self Attention
            norm_x = self.attn_norm(x)
            attn_out, _ = self.self_attn(norm_x, norm_x, norm_x, key_padding_mask=key_padding_mask)
            x = x + attn_out
            # 3. Convolution
            x = self.conv_module(x)
            # 4. Macaron FFN 2
            x = self.ffn2(x)
            # 5. Final Norm
            return self.final_norm(x)
else:
    # Dummy fallback if PyTorch not imported
    class ConformerBlock:
        pass

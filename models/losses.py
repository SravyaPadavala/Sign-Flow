"""
Loss Functions Module
Implements:
1. Connectionist Temporal Classification (CTC) Loss
2. Visual Alignment Constraint (VAC) Loss (Min et al., ICCV 2021)
3. Multi-Task Auxiliary CTC Supervision
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class VisualAlignmentConstraintLoss(nn.Module):
        """
        Visual Alignment Constraint (VAC) Loss.
        Enforces statistical consistency between raw visual backbone outputs
        and sequence model outputs, eliminating the vanishing gradient problem
        in deep Transformer / Conformer backbones.
        """
        def __init__(self, blank_idx=0):
            super().__init__()
            self.blank_idx = blank_idx

        def forward(self, visual_logits, seq_logits, input_lengths):
            """
            Args:
                visual_logits: (B, T, V)
                seq_logits: (B, T, V)
                input_lengths: (B,)
            """
            B, T, V = visual_logits.shape
            
            p_vis = F.softmax(visual_logits, dim=-1)
            log_p_vis = F.log_softmax(visual_logits, dim=-1)
            log_p_seq = F.log_softmax(seq_logits, dim=-1)
            p_seq = F.softmax(seq_logits, dim=-1)
            
            # Symmetrical Kullback-Leibler (KL) divergence
            kl_vis_seq = F.kl_div(log_p_vis, p_seq, reduction='none')
            kl_seq_vis = F.kl_div(log_p_seq, p_vis, reduction='none')
            sym_kl = (kl_vis_seq + kl_seq_vis) * 0.5
            
            loss = 0.0
            total_frames = 0
            for b in range(B):
                l = input_lengths[b]
                loss += sym_kl[b, :l].sum()
                total_frames += l
                
            return loss / (total_frames * V + 1e-6)


    class SignFlowPlusPlusLoss(nn.Module):
        """
        Joint Multi-Task Loss for SignFlow++:
        L_total = L_fused_ctc + lambda_1 * L_aux_visual + lambda_2 * L_aux_pose + lambda_vac * L_vac
        """
        def __init__(self, blank_idx=0, lambda_visual=0.3, lambda_pose=0.3, lambda_vac=0.15):
            super().__init__()
            self.ctc_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
            self.vac_loss = VisualAlignmentConstraintLoss(blank_idx=blank_idx)
            self.lambda_visual = lambda_visual
            self.lambda_pose = lambda_pose
            self.lambda_vac = lambda_vac

        def forward(self, model_outputs, targets, input_lengths, target_lengths):
            """
            Args:
                model_outputs: dict from SignFlowPlusPlusModel
                targets: target tensor of gloss indices
                input_lengths: lengths of input videos
                target_lengths: lengths of gloss sequences
            """
            # PyTorch CTCLoss expects log_probs shape: (T, B, C)
            primary_log_probs = F.log_softmax(model_outputs['logits'], dim=-1).transpose(0, 1)
            loss_primary = self.ctc_loss(primary_log_probs, targets, input_lengths, target_lengths)
            
            aux_vis_log_probs = F.log_softmax(model_outputs['aux_visual_logits'], dim=-1).transpose(0, 1)
            loss_aux_visual = self.ctc_loss(aux_vis_log_probs, targets, input_lengths, target_lengths)
            
            aux_pose_log_probs = F.log_softmax(model_outputs['aux_pose_logits'], dim=-1).transpose(0, 1)
            loss_aux_pose = self.ctc_loss(aux_pose_log_probs, targets, input_lengths, target_lengths)
            
            loss_vac = self.vac_loss(
                model_outputs['aux_visual_logits'],
                model_outputs['logits'],
                input_lengths
            )
            
            total_loss = (loss_primary + 
                          self.lambda_visual * loss_aux_visual + 
                          self.lambda_pose * loss_aux_pose + 
                          self.lambda_vac * loss_vac)
                          
            return {
                'loss': total_loss,
                'loss_primary': loss_primary.item(),
                'loss_aux_visual': loss_aux_visual.item(),
                'loss_aux_pose': loss_aux_pose.item(),
                'loss_vac': loss_vac.item()
            }
else:
    class SignFlowPlusPlusLoss:
        pass

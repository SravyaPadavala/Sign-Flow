"""
CTC Decoding Module
Implements:
1. Greedy CTC Search
2. CTC Prefix Beam Search with length normalization alpha
3. Domain Language Model Rescoring
Designed with pure-Python fallbacks so it operates with or without numpy.
"""

import math
from collections import defaultdict

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

def greedy_ctc_decode(log_probs, blank_idx=0, idx_to_gloss=None):
    """
    Standard Greedy CTC Decoding:
    Selects argmax at each frame, collapses consecutive duplicate tokens, and removes blanks.
    """
    if hasattr(log_probs, 'detach'):
        log_probs = log_probs.detach().cpu().numpy()
        
    decoded = []
    prev_token = None
    
    for frame_scores in log_probs:
        if HAS_NUMPY and isinstance(frame_scores, np.ndarray):
            token = int(np.argmax(frame_scores))
        else:
            token = max(range(len(frame_scores)), key=lambda i: frame_scores[i])
            
        if token != blank_idx and token != prev_token:
            decoded.append(token)
        prev_token = token
        
    if idx_to_gloss is not None:
        return [idx_to_gloss.get(t, f"<UNK_{t}>") for t in decoded]
    return decoded


def ctc_prefix_beam_search(log_probs, beam_size=8, blank_idx=0, alpha=0.8, lm_rescorer=None, beta=0.3, idx_to_gloss=None):
    """
    CTC Prefix Beam Search Decoder with Length Penalty and optional Language Model Rescoring.
    """
    if hasattr(log_probs, 'detach'):
        log_probs = log_probs.detach().cpu().numpy()
        
    T = len(log_probs)
    
    # Beams store: prefix_tuple -> (prob_blank, prob_non_blank)
    beams = {(): (1.0, 0.0)}
    
    for t in range(T):
        frame_log_probs = log_probs[t]
        probs = [math.exp(lp) for lp in frame_log_probs]
        new_beams = defaultdict(lambda: (0.0, 0.0))
        
        # Prune vocab to top candidate indices for computational speed
        sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        top_indices = sorted_indices[:max(beam_size * 2, 20)]
        
        for prefix, (p_b, p_nb) in beams.items():
            p_total = p_b + p_nb
            
            # Case 1: Emission is blank
            p_b_curr, p_nb_curr = new_beams[prefix]
            new_beams[prefix] = (p_b_curr + p_total * probs[blank_idx], p_nb_curr)
            
            # Case 2: Emission is a non-blank token
            for c in top_indices:
                if c == blank_idx:
                    continue
                    
                prob_c = probs[c]
                last_char = prefix[-1] if len(prefix) > 0 else None
                
                if c == last_char:
                    new_prefix = prefix + (c,)
                    nb_curr = new_beams[new_prefix][1]
                    new_beams[new_prefix] = (new_beams[new_prefix][0], nb_curr + p_b * prob_c)
                    
                    b_prev, nb_prev = new_beams[prefix]
                    new_beams[prefix] = (b_prev, nb_prev + p_nb * prob_c)
                else:
                    new_prefix = prefix + (c,)
                    b_curr, nb_curr = new_beams[new_prefix]
                    new_beams[new_prefix] = (b_curr, nb_curr + p_total * prob_c)
                    
        def score_prefix(item):
            prefix, (pb, pnb) = item
            total_prob = pb + pnb
            length = max(1, len(prefix))
            log_prob = math.log(max(total_prob, 1e-30))
            normalized_score = log_prob / (length ** alpha)
            
            if lm_rescorer is not None and len(prefix) > 0:
                lm_score = lm_rescorer(prefix)
                normalized_score += beta * lm_score
                
            return normalized_score
            
        sorted_beams = sorted(new_beams.items(), key=score_prefix, reverse=True)
        beams = dict(sorted_beams[:beam_size])
        
    best_tuple, (p_b, p_nb) = sorted(beams.items(), key=lambda it: (it[1][0] + it[1][1]) / (max(1, len(it[0])) ** alpha), reverse=True)[0]
    
    if idx_to_gloss is not None:
        return [idx_to_gloss.get(t, f"<UNK_{t}>") for t in best_tuple], (p_b + p_nb)
    return list(best_tuple), (p_b + p_nb)

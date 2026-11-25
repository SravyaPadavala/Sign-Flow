"""
Evaluation Metrics Module
Implements:
1. Word Error Rate (WER) with exact Levenshtein alignment breakdown:
   Insertions (I), Deletions (D), Substitutions (S), and Correct (C).
2. Detection Rate metric (Base Paper Equation 3)
3. Sentence-level Accuracy & Latency (FPS)
Designed with pure-Python fallbacks so it runs with or without numpy.
"""

import time

def compute_levenshtein_alignment(ref_words, hyp_words):
    """
    Computes exact Levenshtein distance dynamic programming matrix
    and backtracks to find the exact counts of Insertions, Deletions,
    Substitutions, and Correct matches.
    
    Args:
        ref_words: list of ground truth words/glosses
        hyp_words: list of predicted hypothesis words/glosses
        
    Returns:
        dict: {'substitutions': S, 'deletions': D, 'insertions': I, 'correct': C}
    """
    r_len = len(ref_words)
    h_len = len(hyp_words)
    
    # 2D table for DP: (r_len + 1) x (h_len + 1)
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    
    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j
        
    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # Deletion
                dp[i][j - 1] + 1,       # Insertion
                dp[i - 1][j - 1] + cost # Match or Substitution
            )
            
    # Backtrack
    i = r_len
    j = h_len
    substitutions = 0
    deletions = 0
    insertions = 0
    correct = 0
    
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost == 0:
                    correct += 1
                else:
                    substitutions += 1
                i -= 1
                j -= 1
                continue
                
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insertions += 1
            j -= 1
        else:
            if i > 0:
                deletions += 1
                i -= 1
            elif j > 0:
                insertions += 1
                j -= 1
                
    return {
        'substitutions': substitutions,
        'deletions': deletions,
        'insertions': insertions,
        'correct': correct,
        'total_ref': r_len,
        'total_hyp': h_len
    }


def compute_wer(ref_words, hyp_words):
    """
    Computes Word Error Rate:
    WER = (Insertions + Deletions + Substitutions) / Reference Words
    """
    if len(ref_words) == 0:
        return (1.0 if len(hyp_words) > 0 else 0.0), {'substitutions': 0, 'deletions': 0, 'insertions': len(hyp_words), 'correct': 0, 'total_ref': 0, 'total_hyp': len(hyp_words)}
        
    align = compute_levenshtein_alignment(ref_words, hyp_words)
    errors = align['insertions'] + align['deletions'] + align['substitutions']
    wer = errors / float(align['total_ref'])
    return wer, align


def compute_detection_rate(ref_words, hyp_words):
    """
    Computes Detection Rate (Base Paper Equation 3):
    Detection Rate = (#Words correctly recognized) / (#Words in reference)
    Evaluates unordered keyword identification.
    """
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) == 0 else 0.0
        
    hyp_set = set(hyp_words)
    matched_words = sum(1 for w in ref_words if w in hyp_set)
    return matched_words / float(len(ref_words))


def evaluate_batch(references, hypotheses):
    """
    Evaluates a collection of sentences and computes macro/micro metrics.
    
    Args:
        references: list of reference gloss lists
        hypotheses: list of hypothesis gloss lists
        
    Returns:
        results: dict containing overall WER, Detection Rate, error rates
    """
    total_sub = 0
    total_del = 0
    total_ins = 0
    total_cor = 0
    total_ref = 0
    
    detection_rates = []
    
    for ref, hyp in zip(references, hypotheses):
        align = compute_levenshtein_alignment(ref, hyp)
        total_sub += align['substitutions']
        total_del += align['deletions']
        total_ins += align['insertions']
        total_cor += align['correct']
        total_ref += align['total_ref']
        
        det_rate = compute_detection_rate(ref, hyp)
        detection_rates.append(det_rate)
        
    overall_wer = (total_sub + total_del + total_ins) / float(max(1, total_ref)) * 100.0
    mean_det_rate = (sum(detection_rates) / float(max(1, len(detection_rates)))) * 100.0
    sub_rate = total_sub / float(max(1, total_ref)) * 100.0
    del_rate = total_del / float(max(1, total_ref)) * 100.0
    ins_rate = total_ins / float(max(1, total_ref)) * 100.0
    
    return {
        'WER': round(overall_wer, 2),
        'Detection_Rate': round(mean_det_rate, 2),
        'Substitution_Rate': round(sub_rate, 2),
        'Deletion_Rate': round(del_rate, 2),
        'Insertion_Rate': round(ins_rate, 2),
        'Total_Reference_Words': total_ref,
        'Correct_Words': total_cor
    }

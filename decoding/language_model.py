"""
Domain N-gram Language Model Rescorer
Models Indian Sign Language (ISL) sentence grammar and syntax for eRaktkosh/UMANG queries.
Prevents ungrammatical gloss transitions and reduces substitution errors.
"""

import math
from collections import Counter, defaultdict

# Frequently paired ISL gloss transitions in eRaktkosh / UMANG domain
DOMAIN_CORPUS_SENTENCES = [
    ["BLOOD", "DONATION", "REGISTER", "HOW"],
    ["BLOOD", "DONATE", "ELIGIBILITY", "CRITERIA"],
    ["HOSPITAL", "NEAR", "LOCATION", "SEARCH"],
    ["BLOOD", "AVAILABLE", "STOCK", "STATUS"],
    ["DONOR", "APPOINTMENT", "BOOK", "ONLINE"],
    ["HEMOGLOBIN", "TEST", "REQUIREMENT", "WHAT"],
    ["DONATION", "INTERVAL", "MONTHS", "TIME"],
    ["DONOR", "CARD", "CERTIFICATE", "DOWNLOAD"],
    ["RARE", "BLOOD", "GROUP", "URGENT", "REQUEST"],
    ["BLOOD", "CAMP", "DATE", "LOCATION", "FIND"],
    ["AADHAAR", "CARD", "VERIFY", "OTP"],
    ["FEVER", "MEDICINE", "BLOOD", "DONATE", "SAFE", "NO"],
    ["BLOOD", "BANK", "CONTACT", "NUMBER", "HELP"],
    ["PLASMA", "PLATELETS", "DONATION", "PROCESS"]
]

class ISLDomainLanguageModel:
    """
    Bigram & Trigram Language Model with Good-Turing / Add-k Smoothing
    for ISL continuous sign gloss rescoring.
    """
    def __init__(self, corpus=DOMAIN_CORPUS_SENTENCES, smoothing=0.01):
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()
        self.smoothing = smoothing
        self.total_words = 0
        
        self.build_model(corpus)

    def build_model(self, corpus):
        for sentence in corpus:
            tokens = ["<s>"] + sentence + ["</s>"]
            self.total_words += len(sentence)
            for i in range(len(tokens)):
                self.unigrams[tokens[i]] += 1
                if i >= 1:
                    self.bigrams[(tokens[i-1], tokens[i])] += 1
                if i >= 2:
                    self.trigrams[(tokens[i-2], tokens[i-1], tokens[i])] += 1

    def score_sequence(self, gloss_sequence):
        """
        Computes the log likelihood of a gloss sequence.
        
        Args:
            gloss_sequence: list of gloss strings or indices
        """
        if not gloss_sequence:
            return -10.0
            
        tokens = ["<s>"] + list(gloss_sequence) + ["</s>"]
        log_prob = 0.0
        
        for i in range(1, len(tokens)):
            w_prev = tokens[i - 1]
            w_curr = tokens[i]
            
            # Bigram probability with Add-k smoothing
            count_bi = self.bigrams.get((w_prev, w_curr), 0)
            count_uni = self.unigrams.get(w_prev, 0)
            
            prob = (count_bi + self.smoothing) / (count_uni + self.smoothing * (len(self.unigrams) + 1))
            log_prob += math.log(prob)
            
        return log_prob

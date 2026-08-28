"""
Defense D3: BMX Hybrid Scorer.
Fuses dense cosine similarity scores with an entropy-weighted lexical overlap computation (Eq. 6):
S_D3(x, x̃) = alpha * S_dense(u, v) + (1 - alpha) * S_lex(x, x̃)
"""

import math
import re
from collections import Counter
from typing import Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer
from .base import DefenseOracle

class BMXDefense(DefenseOracle):
    def __init__(self, dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", alpha: float = 0.50, threshold: float = None):
        super().__init__(name="D3_BMX_Hybrid", threshold=threshold)
        self.alpha = alpha
        self.dense_model = SentenceTransformer(dense_model_name)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def _compute_lexical_similarity(self, text_x: str, text_xtilde: str) -> float:
        tokens_x = self._tokenize(text_x)
        tokens_xtilde = self._tokenize(text_xtilde)
        
        if not tokens_x or not tokens_xtilde:
            return 0.0
        
        counts_x = Counter(tokens_x)
        counts_xtilde = Counter(tokens_xtilde)
        
        # Compute term entropy weights: rare domain keywords have higher weight
        all_tokens = tokens_x + tokens_xtilde
        total_terms = len(all_tokens)
        corpus_counts = Counter(all_tokens)
        
        weights = {}
        for token, cnt in corpus_counts.items():
            prob = cnt / total_terms
            # Information entropy weight: -log2(p)
            weights[token] = -math.log2(prob) if prob > 0 else 1.0
            
        # Weighted Jaccard / BM25-style overlap
        common_tokens = set(counts_x.keys()).intersection(set(counts_xtilde.keys()))
        if not common_tokens:
            return 0.0
            
        intersection_weight = sum(min(counts_x[t], counts_xtilde[t]) * weights.get(t, 1.0) for t in common_tokens)
        union_weight = sum(max(counts_x.get(t, 0), counts_xtilde.get(t, 0)) * weights.get(t, 1.0) for t in set(counts_x.keys()).union(set(counts_xtilde.keys())))
        
        lex_sim = (intersection_weight / union_weight) if union_weight > 0 else 0.0
        return float(lex_sim)

    def _compute_similarity(self, source_text: str, candidate_text: str) -> float:
        if not source_text.strip() or not candidate_text.strip():
            return 0.0
        
        # Dense cosine similarity
        emb = self.dense_model.encode([source_text, candidate_text], convert_to_numpy=True, normalize_embeddings=True)
        s_dense = float(np.dot(emb[0], emb[1]))
        s_dense = max(0.0, min(1.0, s_dense))
        
        # Lexical entropy-weighted similarity
        s_lex = self._compute_lexical_similarity(source_text, candidate_text)
        
        # Hybrid fusion
        s_hybrid = self.alpha * s_dense + (1.0 - self.alpha) * s_lex
        return float(s_hybrid)

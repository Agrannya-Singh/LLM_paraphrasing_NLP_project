"""
Defense D1: Sentence-BERT Bi-Encoder.
Collapses documents to fixed-dimensional vectors u, v via mean token pooling over final hidden states,
and evaluates cosine similarity S_D1(x, x̃) = (u^T v) / (||u||_2 ||v||_2).
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from .base import DefenseOracle

class SBERTDefense(DefenseOracle):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", threshold: float = None):
        super().__init__(name="D1_SBERT", threshold=threshold)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def _compute_similarity(self, source_text: str, candidate_text: str) -> float:
        if not source_text.strip() or not candidate_text.strip():
            return 0.0
        
        # Mean token pooled normalized embeddings
        embeddings = self.model.encode([source_text, candidate_text], convert_to_numpy=True, normalize_embeddings=True)
        u, v = embeddings[0], embeddings[1]
        
        # Cosine similarity between normalized vectors is dot product
        cosine_sim = float(np.dot(u, v))
        # Rescale from [-1, 1] to [0, 1] for similarity metric
        normalized_sim = (cosine_sim + 1.0) / 2.0 if cosine_sim < 0 else cosine_sim
        return normalized_sim

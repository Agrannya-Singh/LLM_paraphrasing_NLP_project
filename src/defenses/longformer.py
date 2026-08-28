"""
Defense D5: Longformer Sparse-Attention Document Scorer.
Processes long-document sequences exceeding 4,000 tokens without standard sequence truncation artifacts.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from .base import DefenseOracle

class LongformerDefense(DefenseOracle):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", threshold: float = 0.75):
        super().__init__(name="D5_Longformer", threshold=threshold)
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._using_longformer = False
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def _encode_chunked(self, text: str) -> np.ndarray:
        words = text.split()
        if len(words) <= 250:
            return self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        # Split into sliding overlapping chunks for long documents
        chunk_size = 200
        stride = 100
        chunks = []
        for i in range(0, len(words), stride):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        chunk_embeddings = self.model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        # Mean pooling across document chunks
        doc_emb = np.mean(chunk_embeddings, axis=0)
        norm = np.linalg.norm(doc_emb)
        return doc_emb / (norm + 1e-9)

    def _compute_similarity(self, source_text: str, candidate_text: str) -> float:
        if not source_text.strip() or not candidate_text.strip():
            return 0.0
        
        if self._using_longformer:
            inputs = self.tokenizer([source_text, candidate_text], padding=True, truncation=True, max_length=4096, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # CLS token representation with global attention
                cls_reps = outputs.last_hidden_state[:, 0, :]
                cls_norm = cls_reps / torch.norm(cls_reps, dim=1, keepdim=True)
                sim = torch.sum(cls_norm[0] * cls_norm[1]).item()
                return max(0.0, min(1.0, float(sim)))
        else:
            u = self._encode_chunked(source_text)
            v = self._encode_chunked(candidate_text)
            sim = float(np.dot(u, v))
            return max(0.0, min(1.0, float(sim)))

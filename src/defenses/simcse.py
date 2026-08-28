"""
Defense D2: SimCSE Bi-Encoder.
Builds representations via contrastive learning utilizing an InfoNCE objective over dropout masks.
Evaluates cosine similarity between isotropic latent embeddings.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from .base import DefenseOracle

class SimCSEDefense(DefenseOracle):
    def __init__(self, model_name: str = "princeton-nlp/sup-simcse-bert-base-uncased", threshold: float = None):
        super().__init__(name="D2_SimCSE", threshold=threshold)
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self._using_hf = True
        except Exception as e:
            # Fallback to SentenceTransformer equivalent if needed
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
            self._using_hf = False

    def _compute_similarity(self, source_text: str, candidate_text: str) -> float:
        if not source_text.strip() or not candidate_text.strip():
            return 0.0
        
        if self._using_hf:
            inputs = self.tokenizer([source_text, candidate_text], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # SimCSE standard representation uses [CLS] representation (first token)
                embeddings = outputs.last_hidden_state[:, 0, :]
                # Normalize
                embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
                sim = torch.sum(embeddings[0] * embeddings[1]).item()
        else:
            embeddings = self.model.encode([source_text, candidate_text], convert_to_numpy=True, normalize_embeddings=True)
            sim = float(np.dot(embeddings[0], embeddings[1]))
            
        return max(0.0, min(1.0, float(sim)))

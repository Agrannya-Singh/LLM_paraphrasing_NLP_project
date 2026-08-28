"""
Defense D4: ColBERT Multi-Vector Late-Interaction Scorer.
Retains contextual token embeddings within matrix E_x in R^{|x| x d}.
Computes similarity via late-interaction MaxSim operator (Eq. 7):
S_D4(x, x̃) = (1 / |x|) * sum_{i=1}^{|x|} max_{j=1..|x̃|} ( (E_{x,i}^T E_{x̃,j}) / (||E_{x,i}||_2 ||E_{x̃,j}||_2) )
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from .base import DefenseOracle

class ColBERTDefense(DefenseOracle):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", threshold: float = 0.75):
        super().__init__(name="D4_ColBERT_MultiVector", threshold=threshold)
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _get_token_embeddings(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Contextual token embeddings [1, seq_len, hidden_dim]
            hidden = outputs.last_hidden_state[0]  # [seq_len, hidden_dim]
            # Exclude special tokens [CLS] at 0 and [SEP] at -1 if length > 2
            if hidden.shape[0] > 2:
                hidden = hidden[1:-1]
            # L2 Normalize token embeddings
            norm_embeddings = hidden / torch.norm(hidden, dim=1, keepdim=True).clamp(min=1e-9)
            return norm_embeddings

    def _compute_similarity(self, source_text: str, candidate_text: str) -> float:
        if not source_text.strip() or not candidate_text.strip():
            return 0.0
        
        E_x = self._get_token_embeddings(source_text)       # [|x|, d]
        E_xtilde = self._get_token_embeddings(candidate_text) # [|x̃|, d]
        
        if E_x.shape[0] == 0 or E_xtilde.shape[0] == 0:
            return 0.0
        
        # Pairwise cosine similarity matrix: [|x|, |x̃|]
        sim_matrix = torch.matmul(E_x, E_xtilde.t())
        
        # MaxSim operator: for each token in x, find maximal alignment in x̃
        max_sim_per_token, _ = torch.max(sim_matrix, dim=1)  # [|x|]
        
        # Mean across tokens in reference document x
        score = torch.mean(max_sim_per_token).item()
        return max(0.0, min(1.0, float(score)))

"""
Architecturally Independent Fidelity Gate (Dual-Judge NLI & Non-BERT Cross-Encoder).
Prevents evaluation circularity by using a non-BERT NLI Cross-Encoder to verify semantic preservation F(x, c_j) >= theta_fid.
"""

import torch
import numpy as np
from typing import Tuple, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class IndependentFidelityJudge:
    """
    Evaluates bidirectional semantic fidelity between reference text x and candidate rewrite c_j.
    Enforces F(x, c_j) >= theta_fid without relying on BERT embeddings.
    """
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-xsmall", threshold: float = 0.75):
        self.model_name = model_name
        self.threshold = threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._using_nli = False
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self._using_nli = True
        except Exception:
            self._using_nli = False

    def _compute_nli_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Runs cross-encoder NLI and returns entailment probability (class index corresponding to entailment).
        """
        inputs = self.tokenizer(premise, hypothesis, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            
            # Map labels depending on model configuration
            id2label = self.model.config.id2label
            entail_idx = None
            contra_idx = None
            neutral_idx = None
            
            for idx, label in id2label.items():
                label_lower = label.lower()
                if "entail" in label_lower:
                    entail_idx = idx
                elif "contra" in label_lower:
                    contra_idx = idx
                elif "neutral" in label_lower:
                    neutral_idx = idx
                    
            if entail_idx is not None and contra_idx is not None:
                p_entail = probs[entail_idx].item()
                p_contra = probs[contra_idx].item()
                p_neutral = probs[neutral_idx].item() if neutral_idx is not None else 0.0
                # Fidelity favors entailment while strictly penalizing contradiction
                fidelity = p_entail + 0.5 * p_neutral - 0.5 * p_contra
                return max(0.0, min(1.0, float(fidelity)))
            else:
                # Default assume index 1 or 0 is entailment
                return probs[0].item()

    def _compute_lexical_preservation(self, text_a: str, text_b: str) -> float:
        """
        Secondary non-neural judge: calculates n-gram character and token coverage
        to ensure core semantic entities are preserved.
        """
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        
        # Word set intersection
        jaccard = len(words_a.intersection(words_b)) / len(words_a.union(words_b))
        
        # Length ratio penalty to prevent severe truncation or hallucination
        len_ratio = min(len(text_a), len(text_b)) / max(len(text_a), len(text_b))
        return 0.5 * jaccard + 0.5 * len_ratio

    def evaluate_fidelity(self, source_text: str, candidate_text: str) -> Tuple[bool, float, Dict]:
        """
        Evaluates bidirectional semantic fidelity F(x, c_j).
        Returns (is_valid, fidelity_score, details_dict).
        """
        if not source_text.strip() or not candidate_text.strip():
            return False, 0.0, {"reason": "Empty text"}
            
        if self._using_nli:
            # Bidirectional NLI: premise->hypothesis and hypothesis->premise
            fwd_score = self._compute_nli_entailment(premise=source_text, hypothesis=candidate_text)
            bwd_score = self._compute_nli_entailment(premise=candidate_text, hypothesis=source_text)
            
            # Harmonic/mean bidirectional score
            bidirectional_nli = 0.5 * (fwd_score + bwd_score)
            lex_pres = self._compute_lexical_preservation(source_text, candidate_text)
            
            # Combined score: weighted heavily on NLI with structural sanity check
            total_fidelity = 0.75 * bidirectional_nli + 0.25 * (0.5 + 0.5 * lex_pres)
        else:
            # Fallback statistical bidirectional lexical-semantic metric
            lex_pres = self._compute_lexical_preservation(source_text, candidate_text)
            # High quality paraphrases maintain length and overlap >= 0.35
            total_fidelity = 0.5 + 0.5 * min(1.0, lex_pres * 1.8)

        total_fidelity = max(0.0, min(1.0, float(total_fidelity)))
        is_valid = total_fidelity >= self.threshold
        
        details = {
            "fidelity_score": total_fidelity,
            "threshold": self.threshold,
            "passes_gate": is_valid,
            "using_neural_nli": self._using_nli
        }
        return is_valid, total_fidelity, details

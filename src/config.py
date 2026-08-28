"""
Configuration and Hyperparameters for Adversarial Paraphrase Attack & Defense Evaluation.
Based on the paper: "Retrieval and Similarity-Based Plagiarism Defenses Under Adaptive, Detector-Aware Paraphrase Attacks"
"""

from dataclasses import dataclass, field
from typing import Dict, List
import os

@dataclass
class ExperimentConfig:
    # Universal fallback detection threshold
    detection_threshold: float = 0.75  # fallback tau
    
    # Methodological Enhancement 1: Per-Architecture Calibrated Decision Thresholds (tau_i)
    # Calibrated on a non-plagiarized benign validation set to guarantee 95% Recall / 5% False Positive Rate (FPR)
    calibrated_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "D1_SBERT": 0.78,
        "D2_SimCSE": 0.88,
        "D3_BMX": 0.48,
        "D4_ColBERT": 0.68,
        "D5_Longformer": 0.76
    })
    use_calibrated_thresholds: bool = True

    # Semantic fidelity verification limit (Eq. 2, Eq. 9, Eq. 10)
    fidelity_threshold: float = 0.75   # theta_fid
    
    # Maximum query budget allowed per document pair (Eq. 2, Section IV)
    max_query_budget: int = 50         # B
    
    # LLM Generation parameters (Section IV)
    generation_temperature: float = 0.90
    generation_nucleus_p: float = 0.95
    candidate_pool_size: int = 4       # K candidates sampled per ablation iteration
    top_k_spans: int = 2               # Top influential spans targeted for rewriting
    
    # BMX Hybrid Scorer weight & sensitivity ablation grid
    bmx_alpha: float = 0.50            # Baseline balance between dense similarity and lexical overlap
    bmx_alpha_sweep: List[float] = field(default_factory=lambda: [0.30, 0.50, 0.70, 0.90])
    
    # Statistical validation parameters
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95
    
    # Model Checkpoints (Section IV)
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    simcse_model: str = "princeton-nlp/sup-simcse-bert-base-uncased"
    nli_model: str = "cross-encoder/nli-deberta-v3-xsmall"
    
    # Output directories
    output_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    data_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

config = ExperimentConfig()

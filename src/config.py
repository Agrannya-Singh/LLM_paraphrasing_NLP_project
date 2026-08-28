"""
Configuration and Hyperparameters for Adversarial Paraphrase Attack & Defense Evaluation.
Based on the paper: "Retrieval and Similarity-Based Plagiarism Defenses Under Adaptive, Detector-Aware Paraphrase Attacks"
"""

from dataclasses import dataclass
import os

@dataclass
class ExperimentConfig:
    # Plagiarism classification baseline threshold (Eq. 1)
    detection_threshold: float = 0.75  # tau
    
    # Semantic fidelity verification limit (Eq. 2, Eq. 9, Eq. 10)
    fidelity_threshold: float = 0.75   # theta_fid
    
    # Maximum query budget allowed per document pair (Eq. 2, Section IV)
    max_query_budget: int = 50         # B
    
    # LLM Generation parameters (Section IV)
    generation_temperature: float = 0.90
    generation_nucleus_p: float = 0.95
    candidate_pool_size: int = 4       # K candidates sampled per ablation iteration
    top_k_spans: int = 2               # Top influential spans targeted for rewriting
    
    # BMX Hybrid Scorer weight (Eq. 6)
    bmx_alpha: float = 0.50            # Balance between dense similarity and lexical overlap
    
    # Tier 3 RL Reward Hyperparameters (Eq. 9)
    reward_beta: float = 1.0           # Multiplier scaling baseline fidelity
    reward_gamma: float = 5.0          # Boundary penalty for violating fidelity threshold
    
    # Model Checkpoints (Section IV)
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    simcse_model: str = "princeton-nlp/sup-simcse-bert-base-uncased"
    nli_model: str = "cross-encoder/nli-deberta-v3-xsmall"
    
    # Output directories
    output_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    data_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

config = ExperimentConfig()

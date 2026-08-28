"""
Quantitative Evaluation Metrics for Adversarial Paraphrasing.
Computes ER, FPER (Eq. 10), Mean Query Complexity (MQC), and Spearman Rank Correlation Decay (Delta rho).
"""

from typing import List, Dict
import numpy as np
from scipy.stats import spearmanr
from ..attacks.base import AttackResult

def compute_evasion_rate(results: List[AttackResult]) -> float:
    """
    ER: Percentage of initially flagged documents successfully pushed beneath threshold tau.
    """
    if not results:
        return 0.0
    evasive_count = sum(1 for r in results if r.is_evasive)
    return float(evasive_count / len(results))

def compute_fper(results: List[AttackResult]) -> float:
    """
    FPER (Eq. 10): Percentage of candidates satisfying both evasion criteria and semantic boundary concurrently.
    FPER = (1 / |D_plag|) * sum (I[S(x, x') < tau AND F(x, x') >= theta_fid])
    """
    if not results:
        return 0.0
    fper_count = sum(1 for r in results if r.is_fper)
    return float(fper_count / len(results))

def compute_mean_query_complexity(results: List[AttackResult]) -> float:
    """
    MQC: Average number of black-box oracle queries consumed per document pair.
    """
    if not results:
        return 0.0
    total_queries = sum(r.queries_consumed for r in results)
    return float(total_queries / len(results))

def compute_spearman_rank_decay(initial_scores: List[float], final_scores: List[float]) -> float:
    """
    Calculates Spearman rank correlation decay (Delta rho) between baseline and post-attack scores.
    """
    if len(initial_scores) < 3 or len(final_scores) < 3:
        return 0.0
    # Baseline self-correlation is 1.0; compute correlation between pre and post rankings
    rho, _ = spearmanr(initial_scores, final_scores)
    if np.isnan(rho):
        return 0.0
    # Decay from identity: 1.0 - rho
    decay = 1.0 - rho
    return float(decay)

def compute_full_metrics_summary(results: List[AttackResult]) -> Dict:
    """
    Aggregates full quantitative metrics across attacks.
    """
    if not results:
        return {}
    
    er = compute_evasion_rate(results)
    fper = compute_fper(results)
    mqc = compute_mean_query_complexity(results)
    
    init_scores = [r.initial_score for r in results]
    final_scores = [r.final_score for r in results]
    delta_rho = compute_spearman_rank_decay(init_scores, final_scores)
    
    avg_fid = float(np.mean([r.fidelity_score for r in results]))
    avg_score_drop = float(np.mean([r.initial_score - r.final_score for r in results]))

    return {
        "total_samples": len(results),
        "evasion_rate_er": er,
        "fidelity_preserved_evasion_rate_fper": fper,
        "mean_query_complexity_mqc": mqc,
        "rank_correlation_decay_delta_rho": delta_rho,
        "average_fidelity_score": avg_fid,
        "average_score_drop": avg_score_drop
    }

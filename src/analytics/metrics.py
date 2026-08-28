"""
Quantitative Evaluation Metrics & Statistical Significance for Adversarial Paraphrasing.
Computes:
1. Core Metrics: ER, FPER (Eq. 10), Mean Query Complexity (MQC), and Spearman Rank Decay (Delta rho)
2. Statistical Rigor: 95% Bootstrap Confidence Intervals & Wilcoxon Signed-Rank Test p-values
"""

from typing import List, Dict, Tuple
import numpy as np
from scipy.stats import spearmanr, wilcoxon, ttest_rel
from ..attacks.base import AttackResult

def compute_evasion_rate(results: List[AttackResult]) -> float:
    """
    ER: Percentage of initially flagged documents successfully pushed beneath calibrated threshold tau_i.
    """
    if not results:
        return 0.0
    evasive_count = sum(1 for r in results if r.is_evasive)
    return float(evasive_count / len(results))

def compute_fper(results: List[AttackResult]) -> float:
    """
    FPER (Eq. 10): Percentage of candidates satisfying both evasion criteria and semantic boundary concurrently.
    FPER = (1 / |D_plag|) * sum (I[S(x, x') < tau_i AND F(x, x') >= theta_fid])
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
    rho, _ = spearmanr(initial_scores, final_scores)
    if np.isnan(rho):
        return 0.0
    decay = 1.0 - rho
    return float(decay)

def compute_bootstrap_ci(values: List[float], n_bootstraps: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    """
    Computes empirical mean and [low, high] non-parametric bootstrap confidence interval.
    """
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values)
    mean_val = float(np.mean(arr))
    if len(arr) < 2:
        return mean_val, mean_val, mean_val
        
    rng = np.random.default_rng(seed=42)
    boot_means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_bootstraps)]
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return mean_val, low, high

def compute_significance_test(initial_scores: List[float], final_scores: List[float]) -> Dict[str, float]:
    """
    Computes Wilcoxon signed-rank and Paired t-test p-values to prove score drops (s0 -> s') are statistically significant.
    """
    if len(initial_scores) < 5 or len(final_scores) < 5:
        return {"p_value_wilcoxon": 1.0, "p_value_ttest": 1.0, "is_statistically_significant": False}
        
    diffs = np.array(initial_scores) - np.array(final_scores)
    
    # Wilcoxon signed-rank test
    try:
        w_stat, p_wilcoxon = wilcoxon(diffs, alternative="greater")
        p_w = float(p_wilcoxon)
    except Exception:
        p_w = 1.0
        
    # Paired Student's t-test
    try:
        t_stat, p_t = ttest_rel(initial_scores, final_scores, alternative="greater")
        p_tt = float(p_t)
    except Exception:
        p_tt = 1.0
        
    return {
        "p_value_wilcoxon": p_w,
        "p_value_ttest": p_tt,
        "is_statistically_significant": bool(p_w < 0.05 or p_tt < 0.05)
    }

def compute_full_metrics_summary(results: List[AttackResult]) -> Dict:
    """
    Aggregates full quantitative metrics with standard deviations, 95% bootstrap CIs, and p-values.
    """
    if not results:
        return {}
    
    er = compute_evasion_rate(results)
    fper = compute_fper(results)
    mqc = compute_mean_query_complexity(results)
    
    fper_flags = [1.0 if r.is_fper else 0.0 for r in results]
    _, fper_ci_low, fper_ci_high = compute_bootstrap_ci(fper_flags)
    
    queries = [float(r.queries_consumed) for r in results]
    _, mqc_ci_low, mqc_ci_high = compute_bootstrap_ci(queries)
    
    init_scores = [r.initial_score for r in results]
    final_scores = [r.final_score for r in results]
    delta_rho = compute_spearman_rank_decay(init_scores, final_scores)
    
    avg_fid = float(np.mean([r.fidelity_score for r in results]))
    std_fid = float(np.std([r.fidelity_score for r in results]))
    
    score_drops = [r.initial_score - r.final_score for r in results]
    avg_score_drop = float(np.mean(score_drops))
    std_score_drop = float(np.std(score_drops))
    
    sig_test = compute_significance_test(init_scores, final_scores)

    return {
        "total_samples": len(results),
        "evasion_rate_er": er,
        "fidelity_preserved_evasion_rate_fper": fper,
        "fper_95_ci": [fper_ci_low, fper_ci_high],
        "mean_query_complexity_mqc": mqc,
        "mqc_95_ci": [mqc_ci_low, mqc_ci_high],
        "rank_correlation_decay_delta_rho": delta_rho,
        "average_fidelity_score": avg_fid,
        "std_fidelity_score": std_fid,
        "average_score_drop": avg_score_drop,
        "std_score_drop": std_score_drop,
        "statistical_significance": sig_test
    }

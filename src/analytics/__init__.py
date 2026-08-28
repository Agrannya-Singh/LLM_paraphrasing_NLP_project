"""
Analytics and Evaluation Package
"""
from .metrics import (
    compute_evasion_rate,
    compute_fper,
    compute_mean_query_complexity,
    compute_spearman_rank_decay,
    compute_full_metrics_summary
)
from .transferability import compute_transferability_matrix
from .reporter import ExperimentReporter

__all__ = [
    "compute_evasion_rate",
    "compute_fper",
    "compute_mean_query_complexity",
    "compute_spearman_rank_decay",
    "compute_full_metrics_summary",
    "compute_transferability_matrix",
    "ExperimentReporter"
]

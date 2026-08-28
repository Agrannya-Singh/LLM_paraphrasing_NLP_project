"""
Abstract Base Defense Oracle.
Defines the black-box interface, query tracker, and calibrated binary detection threshold evaluator.
"""

from abc import ABC, abstractmethod
from typing import Tuple
from src.config import config

class DefenseOracle(ABC):
    """
    Abstract Base Class for Black-Box Continuous Similarity Plagiarism Defenses.
    Supports both universal and independently calibrated decision thresholds (tau_i).
    """
    def __init__(self, name: str, threshold: float = None):
        self.name = name
        if threshold is None:
            if config.use_calibrated_thresholds:
                matched_tau = None
                for key, val in config.calibrated_thresholds.items():
                    if key == name or key in name or name in key:
                        matched_tau = val
                        break
                self.threshold = matched_tau if matched_tau is not None else config.detection_threshold
            else:
                self.threshold = config.detection_threshold
        else:
            self.threshold = threshold
        self.query_count = 0

    def reset_query_count(self):
        self.query_count = 0

    @abstractmethod
    def _compute_similarity(self, source_text: str, candidate_text: str) -> float:
        """
        Internal implementation of continuous scalar similarity S(x, x̃).
        Must return a float bounded in [0.0, 1.0].
        """
        pass

    def score(self, source_text: str, candidate_text: str) -> float:
        """
        Public query interface. Increments query counter and returns scalar score S(x, x̃).
        """
        self.query_count += 1
        raw_score = self._compute_similarity(source_text, candidate_text)
        # Ensure clamped to [0, 1]
        clamped_score = max(0.0, min(1.0, float(raw_score)))
        return clamped_score

    def classify(self, source_text: str, candidate_text: str) -> Tuple[bool, float]:
        """
        Evaluates binary decision ŷ = I[S(x, x̃) >= tau_i] and returns (is_plagiarized, similarity_score).
        """
        score = self.score(source_text, candidate_text)
        is_plag = score >= self.threshold
        return is_plag, score

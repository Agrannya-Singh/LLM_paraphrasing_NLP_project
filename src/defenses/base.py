"""
Abstract Base Defense Oracle.
Defines the black-box interface, query tracker, and binary detection threshold evaluator.
"""

from abc import ABC, abstractmethod
from typing import Tuple

class DefenseOracle(ABC):
    """
    Abstract Base Class for Black-Box Continuous Similarity Plagiarism Defenses.
    """
    def __init__(self, name: str, threshold: float = 0.75):
        self.name = name
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
        Evaluates binary decision ŷ = I[S(x, x̃) >= tau] and returns (is_plagiarized, similarity_score).
        """
        score = self.score(source_text, candidate_text)
        is_plag = score >= self.threshold
        return is_plag, score

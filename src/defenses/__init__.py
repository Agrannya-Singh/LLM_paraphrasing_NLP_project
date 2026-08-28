"""
Defense Oracles Package (D1 to D5)
"""

from .base import DefenseOracle
from .sbert import SBERTDefense
from .simcse import SimCSEDefense
from .bmx import BMXDefense
from .colbert import ColBERTDefense
from .longformer import LongformerDefense

def load_defense_matrix(threshold: float = 0.75):
    """
    Instantiates the five-architecture defense matrix D1 to D5.
    """
    return {
        "D1_SBERT": SBERTDefense(threshold=threshold),
        "D2_SimCSE": SimCSEDefense(threshold=threshold),
        "D3_BMX": BMXDefense(threshold=threshold),
        "D4_ColBERT": ColBERTDefense(threshold=threshold),
        "D5_Longformer": LongformerDefense(threshold=threshold),
    }

__all__ = [
    "DefenseOracle",
    "SBERTDefense",
    "SimCSEDefense",
    "BMXDefense",
    "ColBERTDefense",
    "LongformerDefense",
    "load_defense_matrix"
]

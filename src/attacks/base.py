"""
Base Attack Engine and Result Structures.
Implements the budget monitor (b <= B) and standardized trajectory tracking.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from ..defenses.base import DefenseOracle
from ..fidelity.judge import IndependentFidelityJudge

@dataclass
class AttackStep:
    step_idx: int
    current_score: float
    query_count_consumed: int
    action_type: str
    action_detail: Dict
    candidate_text: str
    fidelity_score: float
    is_fper: bool

@dataclass
class AttackResult:
    pair_id: str
    domain: str
    defense_name: str
    attack_tier: str
    source_text: str
    initial_suspect_text: str
    final_paraphrase_text: str
    initial_score: float
    final_score: float
    threshold: float
    fidelity_score: float
    fidelity_threshold: float
    is_evasive: bool           # S(x, x') < tau
    passes_fidelity: bool      # F(x, x') >= theta_fid
    is_fper: bool              # S(x, x') < tau AND F(x, x') >= theta_fid
    queries_consumed: int
    budget: int
    trajectory: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

class BaseAttack(ABC):
    def __init__(self, budget: int = 50, fidelity_threshold: float = 0.75):
        self.budget = budget
        self.fidelity_threshold = fidelity_threshold

    @abstractmethod
    def execute(self,
                source_text: str,
                suspect_text: str,
                defense: DefenseOracle,
                fidelity_judge: IndependentFidelityJudge,
                pair_id: str = "PAIR_0",
                domain: str = "general") -> AttackResult:
        pass

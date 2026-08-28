"""
Tier 1: Static Paraphrase Attack (Zero-Feedback Baseline).
Evaluates single-pass LLM rewrite performance without iterative detector interaction.
"""

from .base import BaseAttack, AttackResult, AttackStep
from .llm_generator import LLMParaphraseGenerator
from ..defenses.base import DefenseOracle
from ..fidelity.judge import IndependentFidelityJudge

class Tier1StaticAttack(BaseAttack):
    def __init__(self, budget: int = 50, fidelity_threshold: float = 0.75, generator: LLMParaphraseGenerator = None):
        super().__init__(budget=budget, fidelity_threshold=fidelity_threshold)
        self.generator = generator or LLMParaphraseGenerator()

    def execute(self,
                source_text: str,
                suspect_text: str,
                defense: DefenseOracle,
                fidelity_judge: IndependentFidelityJudge,
                pair_id: str = "PAIR_0",
                domain: str = "general") -> AttackResult:
        
        # 1. Measure initial score s0
        initial_score = defense.score(source_text, suspect_text)
        
        # If already below threshold, immediate return
        if initial_score < defense.threshold:
            _, fid_score, _ = fidelity_judge.evaluate_fidelity(source_text, suspect_text)
            return AttackResult(
                pair_id=pair_id,
                domain=domain,
                defense_name=defense.name,
                attack_tier="Tier1_Static",
                source_text=source_text,
                initial_suspect_text=suspect_text,
                final_paraphrase_text=suspect_text,
                initial_score=initial_score,
                final_score=initial_score,
                threshold=defense.threshold,
                fidelity_score=fid_score,
                fidelity_threshold=self.fidelity_threshold,
                is_evasive=True,
                passes_fidelity=(fid_score >= self.fidelity_threshold),
                is_fper=(fid_score >= self.fidelity_threshold),
                queries_consumed=1,
                budget=self.budget,
                trajectory=[]
            )

        # 2. Generate static paraphrase
        paraphrase_text = self.generator.generate_static_paraphrase(suspect_text)
        
        # 3. Query defense score
        final_score = defense.score(source_text, paraphrase_text)
        
        # 4. Verify semantic fidelity
        passes_fid, fid_score, _ = fidelity_judge.evaluate_fidelity(source_text, paraphrase_text)
        
        is_evasive = final_score < defense.threshold
        is_fper = is_evasive and passes_fid

        trajectory = [{
            "step_idx": 1,
            "action": "static_rewrite",
            "candidate_text": paraphrase_text,
            "similarity_score": final_score,
            "fidelity_score": fid_score,
            "is_evasive": is_evasive,
            "is_fper": is_fper,
            "queries_consumed": 2
        }]

        return AttackResult(
            pair_id=pair_id,
            domain=domain,
            defense_name=defense.name,
            attack_tier="Tier1_Static",
            source_text=source_text,
            initial_suspect_text=suspect_text,
            final_paraphrase_text=paraphrase_text,
            initial_score=initial_score,
            final_score=final_score,
            threshold=defense.threshold,
            fidelity_score=fid_score,
            fidelity_threshold=self.fidelity_threshold,
            is_evasive=is_evasive,
            passes_fidelity=passes_fid,
            is_fper=is_fper,
            queries_consumed=2,
            budget=self.budget,
            trajectory=trajectory
        )

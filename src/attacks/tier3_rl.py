"""
Tier 3: Closed-Loop RL Policy Formulation & Optimization.
Optimizes candidate perturbations against a scalar reward function coupling evasion progress with a hard semantic penalty (Eq. 9):
R(x, x') = -S_D(x, x') + beta * F(x, x') - gamma * max(0, theta_fid - F(x, x'))
"""

from typing import List, Dict
from .base import BaseAttack, AttackResult
from .llm_generator import LLMParaphraseGenerator
from ..corpus.segmenter import SyntacticSegmenter
from ..defenses.base import DefenseOracle
from ..fidelity.judge import IndependentFidelityJudge

class Tier3RLPolicyAttack(BaseAttack):
    def __init__(self,
                 budget: int = 50,
                 fidelity_threshold: float = 0.75,
                 reward_beta: float = 1.0,
                 reward_gamma: float = 5.0,
                 generator: LLMParaphraseGenerator = None):
        super().__init__(budget=budget, fidelity_threshold=fidelity_threshold)
        self.beta = reward_beta
        self.gamma = reward_gamma
        self.segmenter = SyntacticSegmenter()
        self.generator = generator or LLMParaphraseGenerator()

    def compute_reward(self, score: float, fidelity: float) -> float:
        """
        Scalar reward function R(x, x') = -S_D(x, x') + beta * F(x, x') - gamma * max(0, theta_fid - F(x, x'))
        """
        penalty = max(0.0, self.fidelity_threshold - fidelity)
        reward = -score + (self.beta * fidelity) - (self.gamma * penalty)
        return float(reward)

    def execute(self,
                source_text: str,
                suspect_text: str,
                defense: DefenseOracle,
                fidelity_judge: IndependentFidelityJudge,
                pair_id: str = "PAIR_0",
                domain: str = "general") -> AttackResult:
        
        queries_used = 0
        trajectory = []
        
        # 1. Baseline initialization
        current_text = suspect_text
        current_score = defense.score(source_text, current_text)
        queries_used += 1
        initial_score = current_score
        
        _, current_fid, _ = fidelity_judge.evaluate_fidelity(source_text, current_text)
        current_reward = self.compute_reward(current_score, current_fid)
        
        best_text = current_text
        best_score = current_score
        best_fid = current_fid
        best_reward = current_reward
        
        step_idx = 0
        while queries_used < self.budget and (best_score >= defense.threshold or best_fid < self.fidelity_threshold):
            step_idx += 1
            spans = self.segmenter.segment(current_text)
            if not spans:
                break
                
            # Sample diverse action proposals across span perturbation mutations
            candidate_actions = []
            for sp_idx in range(len(spans)):
                candidates = self.generator.generate_targeted_candidates(
                    full_text=current_text,
                    spans=spans,
                    target_span_indices=[sp_idx],
                    pool_size=2
                )
                candidate_actions.extend(candidates)
                
            # Deduplicate
            unique_candidates = list(set(candidate_actions))
            
            # Policy evaluation
            scored_candidates = []
            for cand in unique_candidates:
                if queries_used >= self.budget:
                    break
                s = defense.score(source_text, cand)
                queries_used += 1
                _, f, _ = fidelity_judge.evaluate_fidelity(source_text, cand)
                r = self.compute_reward(s, f)
                scored_candidates.append((cand, s, f, r))
                
            if not scored_candidates:
                break
                
            # Policy argmax over reward R(x, x')
            scored_candidates.sort(key=lambda x: x[3], reverse=True)
            chosen_cand, chosen_s, chosen_f, chosen_r = scored_candidates[0]
            
            trajectory.append({
                "step_idx": step_idx,
                "action": "rl_reward_maximization",
                "reward": chosen_r,
                "similarity_score": chosen_s,
                "fidelity_score": chosen_f,
                "queries_used": queries_used,
                "is_evasive": bool(chosen_s < defense.threshold),
                "is_fper": bool(chosen_s < defense.threshold and chosen_f >= self.fidelity_threshold)
            })
            
            if chosen_r > best_reward:
                best_reward = chosen_r
                best_text = chosen_cand
                best_score = chosen_s
                best_fid = chosen_f
                current_text = chosen_cand
                
            if best_score < defense.threshold and best_fid >= self.fidelity_threshold:
                break

        is_evasive = best_score < defense.threshold
        passes_fidelity = best_fid >= self.fidelity_threshold
        is_fper = is_evasive and passes_fidelity

        return AttackResult(
            pair_id=pair_id,
            domain=domain,
            defense_name=defense.name,
            attack_tier="Tier3_RL_Policy",
            source_text=source_text,
            initial_suspect_text=suspect_text,
            final_paraphrase_text=best_text,
            initial_score=initial_score,
            final_score=best_score,
            threshold=defense.threshold,
            fidelity_score=best_fid,
            fidelity_threshold=self.fidelity_threshold,
            is_evasive=is_evasive,
            passes_fidelity=passes_fidelity,
            is_fper=is_fper,
            queries_consumed=queries_used,
            budget=self.budget,
            trajectory=trajectory
        )

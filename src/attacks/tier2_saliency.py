"""
Tier 2: Saliency-Guided Adaptive Paraphrase Attack with Rejection Sampling.
Combines Leave-One-Out (LOO) span ablation with feedback-driven rejection sampling and fidelity gating (Figures 2 & 3).
"""

from typing import List, Dict, Tuple
from .base import BaseAttack, AttackResult, AttackStep
from .llm_generator import LLMParaphraseGenerator
from ..corpus.segmenter import SyntacticSegmenter
from ..defenses.base import DefenseOracle
from ..fidelity.judge import IndependentFidelityJudge

class Tier2SaliencyAttack(BaseAttack):
    def __init__(self,
                 budget: int = 50,
                 fidelity_threshold: float = 0.75,
                 top_k_spans: int = 2,
                 candidate_pool_size: int = 4,
                 generator: LLMParaphraseGenerator = None):
        super().__init__(budget=budget, fidelity_threshold=fidelity_threshold)
        self.top_k_spans = top_k_spans
        self.candidate_pool_size = candidate_pool_size
        self.segmenter = SyntacticSegmenter()
        self.generator = generator or LLMParaphraseGenerator()

    def execute(self,
                source_text: str,
                suspect_text: str,
                defense: DefenseOracle,
                fidelity_judge: IndependentFidelityJudge,
                pair_id: str = "PAIR_0",
                domain: str = "general") -> AttackResult:
        
        queries_used = 0
        trajectory = []
        
        # 1. Check baseline initial score s0
        current_text = suspect_text
        current_score = defense.score(source_text, current_text)
        queries_used += 1
        
        initial_score = current_score
        passes_fid, current_fid, _ = fidelity_judge.evaluate_fidelity(source_text, current_text)

        # If already below threshold
        if current_score < defense.threshold and passes_fid:
            return AttackResult(
                pair_id=pair_id,
                domain=domain,
                defense_name=defense.name,
                attack_tier="Tier2_Saliency",
                source_text=source_text,
                initial_suspect_text=suspect_text,
                final_paraphrase_text=current_text,
                initial_score=initial_score,
                final_score=current_score,
                threshold=defense.threshold,
                fidelity_score=current_fid,
                fidelity_threshold=self.fidelity_threshold,
                is_evasive=True,
                passes_fidelity=True,
                is_fper=True,
                queries_consumed=queries_used,
                budget=self.budget,
                trajectory=[]
            )

        step_idx = 0
        best_candidate = current_text
        best_score = current_score
        best_fid = current_fid

        # Iterative optimization loop
        while queries_used < self.budget and best_score >= defense.threshold:
            step_idx += 1
            
            # Phase A: Syntactic Span Partitioning & Leave-One-Out (LOO) Saliency Ablation
            spans = self.segmenter.segment(current_text)
            if not spans:
                break
                
            attributions = []
            for i, span in enumerate(spans):
                if queries_used >= self.budget:
                    break
                ablated_text = self.segmenter.ablate_span(spans, i)
                ablated_score = defense.score(source_text, ablated_text)
                queries_used += 1
                
                # Attribution metric: I(w_i) = |S(x, x̃) - S(x, x̃ \ w_i)| (Eq. 8)
                importance = abs(current_score - ablated_score)
                attributions.append((i, importance, span))

            if not attributions:
                break

            # Sort spans by attribution score descending
            attributions.sort(key=lambda x: x[1], reverse=True)
            top_span_indices = [item[0] for item in attributions[:self.top_k_spans]]

            # Phase B: Targeted Rewrite Generation for influential spans
            raw_candidates = self.generator.generate_targeted_candidates(
                full_text=current_text,
                spans=spans,
                target_span_indices=top_span_indices,
                pool_size=self.candidate_pool_size
            )

            # Phase C: Semantic Filtering (Architecturally Independent Gate)
            valid_candidates = []
            for cand in raw_candidates:
                cand_passes, cand_fid, _ = fidelity_judge.evaluate_fidelity(source_text, cand)
                if cand_passes:
                    valid_candidates.append((cand, cand_fid))

            # Fallback if no candidate passed strict fidelity gate
            if not valid_candidates:
                # Use highest scoring raw candidate or current text
                fallback_cand = raw_candidates[0] if raw_candidates else current_text
                _, cand_fid, _ = fidelity_judge.evaluate_fidelity(source_text, fallback_cand)
                valid_candidates.append((fallback_cand, cand_fid))

            # Phase D: Rejection Sampling & Selection
            scored_candidates = []
            for cand_text, cand_fid in valid_candidates:
                if queries_used >= self.budget:
                    break
                c_score = defense.score(source_text, cand_text)
                queries_used += 1
                scored_candidates.append((cand_text, c_score, cand_fid))

            if not scored_candidates:
                break

            # Select candidate with minimal continuous similarity score: min S(x, c_j)
            scored_candidates.sort(key=lambda x: x[1])
            chosen_cand, chosen_score, chosen_fid = scored_candidates[0]

            # Update current state
            step_record = {
                "step_idx": step_idx,
                "top_ablated_spans": [item[2] for item in attributions[:self.top_k_spans]],
                "attribution_scores": [float(item[1]) for item in attributions[:self.top_k_spans]],
                "num_valid_candidates": len(valid_candidates),
                "chosen_score": float(chosen_score),
                "chosen_fidelity": float(chosen_fid),
                "queries_used_so_far": queries_used,
                "is_evasive": bool(chosen_score < defense.threshold),
                "is_fper": bool(chosen_score < defense.threshold and chosen_fid >= self.fidelity_threshold)
            }
            trajectory.append(step_record)

            if chosen_score < best_score:
                best_score = chosen_score
                best_candidate = chosen_cand
                best_fid = chosen_fid
                current_text = chosen_cand
                current_score = chosen_score

            # If successful evasion with fidelity preservation, terminate early
            if best_score < defense.threshold and best_fid >= self.fidelity_threshold:
                break

        is_evasive = best_score < defense.threshold
        passes_fidelity = best_fid >= self.fidelity_threshold
        is_fper = is_evasive and passes_fidelity

        return AttackResult(
            pair_id=pair_id,
            domain=domain,
            defense_name=defense.name,
            attack_tier="Tier2_Saliency",
            source_text=source_text,
            initial_suspect_text=suspect_text,
            final_paraphrase_text=best_candidate,
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

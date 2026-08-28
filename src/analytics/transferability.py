"""
Cross-Architecture Transferability Matrix & Normalized Transferability Delta Computation.
Computes:
1. Raw Transferability Matrix: T_{i,j} = FPER(P_{D_i}(x̃) -> D_j)
2. Normalized Transferability Delta: Delta T_{i,j} = T_{i,j} - FPER(Static Baseline -> D_j)
   (Isolates true adversarial distortion from static baseline sensitivity)
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from ..defenses.base import DefenseOracle
from ..fidelity.judge import IndependentFidelityJudge
from ..attacks.base import AttackResult

def compute_transferability_matrix(optimized_results_by_defense: Dict[str, List[AttackResult]],
                                  defense_matrix: Dict[str, DefenseOracle],
                                  fidelity_judge: IndependentFidelityJudge,
                                  static_baseline_results: List[AttackResult] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constructs:
    - Raw 5x5 Cross-Architecture Transferability Matrix T in R^{5x5}.
    - Normalized Transferability Delta Matrix Delta T in R^{5x5}.
    
    Rows (i): Source Defense used during adversarial optimization.
    Columns (j): Target Defense evaluated.
    """
    defense_names = list(defense_matrix.keys())
    raw_matrix = np.zeros((len(defense_names), len(defense_names)), dtype=float)
    
    # 1. Compute Raw Transferability Matrix T_{i, j}
    for i, src_name in enumerate(defense_names):
        src_results = optimized_results_by_defense.get(src_name, [])
        if not src_results:
            continue
            
        for j, tgt_name in enumerate(defense_names):
            tgt_defense = defense_matrix[tgt_name]
            
            fper_hits = 0
            for r in src_results:
                source_text = r.source_text
                optimized_candidate = r.final_paraphrase_text
                
                # Forward pass against target defense
                tgt_score = tgt_defense.score(source_text, optimized_candidate)
                # Verify fidelity
                passes_fid, fid_score, _ = fidelity_judge.evaluate_fidelity(source_text, optimized_candidate)
                
                # Check FPER condition against target defense threshold (calibrated tau_j)
                if tgt_score < tgt_defense.threshold and passes_fid:
                    fper_hits += 1
                    
            raw_matrix[i, j] = fper_hits / len(src_results) if src_results else 0.0

    df_raw = pd.DataFrame(raw_matrix, index=defense_names, columns=defense_names)

    # 2. Compute Baseline Sensitivity per Target: FPER(Static -> D_j)
    baseline_fper = np.zeros(len(defense_names), dtype=float)
    if static_baseline_results:
        for j, tgt_name in enumerate(defense_names):
            tgt_defense = defense_matrix[tgt_name]
            hits = 0
            for r in static_baseline_results:
                score = tgt_defense.score(r.source_text, r.final_paraphrase_text)
                passes, _, _ = fidelity_judge.evaluate_fidelity(r.source_text, r.final_paraphrase_text)
                if score < tgt_defense.threshold and passes:
                    hits += 1
            baseline_fper[j] = hits / len(static_baseline_results) if static_baseline_results else 0.0
    else:
        # Fallback: estimate baseline from the diagonal min
        baseline_fper = np.min(raw_matrix, axis=0) * 0.85

    # 3. Compute Normalized Transferability Delta: Delta T_{i, j} = T_{i, j} - Baseline_j
    delta_matrix = raw_matrix - baseline_fper.reshape(1, -1)
    df_delta = pd.DataFrame(delta_matrix, index=defense_names, columns=defense_names)

    return df_raw, df_delta

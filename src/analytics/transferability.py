"""
Cross-Architecture Transferability Matrix Computation.
Computes T_{i,j} = FPER(P_{D_i}(x̃) -> D_j) for all pairs of defense architectures (Eq. 11).
"""

from typing import Dict, List
import pandas as pd
import numpy as np
from ..defenses.base import DefenseOracle
from ..fidelity.judge import IndependentFidelityJudge
from ..attacks.base import AttackResult

def compute_transferability_matrix(optimized_results_by_defense: Dict[str, List[AttackResult]],
                                  defense_matrix: Dict[str, DefenseOracle],
                                  fidelity_judge: IndependentFidelityJudge) -> pd.DataFrame:
    """
    Constructs 5x5 Cross-Architecture Transferability Matrix T in R^{5x5}.
    Rows (i): Source Defense used during adversarial optimization.
    Columns (j): Target Defense evaluated.
    """
    defense_names = list(defense_matrix.keys())
    matrix = np.zeros((len(defense_names), len(defense_names)), dtype=float)
    
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
                
                # Check FPER condition against target defense threshold
                if tgt_score < tgt_defense.threshold and passes_fid:
                    fper_hits += 1
                    
            matrix[i, j] = fper_hits / len(src_results) if src_results else 0.0

    df = pd.DataFrame(matrix, index=defense_names, columns=defense_names)
    return df

"""
End-to-End Adversarial Plagiarism Defense Evaluation Pipeline (Static & Adaptive Evaluation).
Runs evaluation across 5 domains, 5 defense oracles, and multiple paraphrase attack strategies.
Supports benchmarking static corpora (including Claude/GPT pre-populated paraphrases) and adaptive search.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List

# Ensure line buffering on stdout
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import config
from src.corpus.loader import CorpusLoader
from src.defenses import load_defense_matrix
from src.fidelity.judge import IndependentFidelityJudge
from src.attacks.llm_generator import LLMParaphraseGenerator
from src.attacks.tier1_static import Tier1StaticAttack
from src.attacks.tier2_saliency import Tier2SaliencyAttack
from src.attacks.tier3_rl import Tier3RLPolicyAttack
from src.attacks.base import AttackResult
from src.analytics.metrics import compute_full_metrics_summary
from src.analytics.transferability import compute_transferability_matrix
from src.analytics.reporter import ExperimentReporter

def evaluate_static_paraphrase_candidate(
    pair_id: str,
    domain: str,
    source_text: str,
    suspect_text: str,
    candidate_text: str,
    strategy_name: str,
    defense,
    fidelity_judge,
    budget: int = 50
) -> AttackResult:
    """
    Evaluates a pre-generated static paraphrase (e.g. from Claude, GPT, or adversarial corpus)
    against a target defense oracle and the independent fidelity gate.
    """
    initial_score = defense.score(source_text, suspect_text)
    final_score = defense.score(source_text, candidate_text)
    passes_fid, fid_score, _ = fidelity_judge.evaluate_fidelity(source_text, candidate_text)

    is_evasive = final_score < defense.threshold
    is_fper = is_evasive and passes_fid

    return AttackResult(
        pair_id=pair_id,
        domain=domain,
        defense_name=defense.name,
        attack_tier=strategy_name,
        source_text=source_text,
        initial_suspect_text=suspect_text,
        final_paraphrase_text=candidate_text,
        initial_score=initial_score,
        final_score=final_score,
        threshold=defense.threshold,
        fidelity_score=fid_score,
        fidelity_threshold=fidelity_judge.threshold,
        is_evasive=is_evasive,
        passes_fidelity=passes_fid,
        is_fper=is_fper,
        queries_consumed=2,
        budget=budget,
        trajectory=[{
            "step_idx": 1,
            "action": strategy_name,
            "similarity_score": final_score,
            "fidelity_score": fid_score,
            "is_evasive": is_evasive,
            "is_fper": is_fper
        }]
    )

def run_evaluation_pipeline(num_pairs: int = None, domain_filter: str = None):
    print("=" * 85)
    print("ADAPTIVE DETECTOR-AWARE PARAPHRASE ATTACK & DEFENSE EVALUATION")
    print("Paper: 'Retrieval and Similarity-Based Plagiarism Defenses Under Adaptive, Detector-Aware Paraphrase Attacks'")
    print("=" * 85, flush=True)

    # 1. Load Data
    print("\n[Stage 1] Loading Corpus & Benchmark Pairs...", flush=True)
    loader = CorpusLoader()
    all_pairs = loader.get_all_pairs()
    
    if domain_filter:
        pairs = [p for p in all_pairs if p.domain.lower() == domain_filter.lower()]
    else:
        pairs = all_pairs

    if num_pairs and num_pairs < len(pairs):
        # Sample evenly across available domains
        from collections import defaultdict
        by_dom = defaultdict(list)
        for p in pairs:
            by_dom[p.domain].append(p)
        sampled = []
        per_dom = max(1, num_pairs // len(by_dom))
        for dom, p_list in by_dom.items():
            sampled.extend(p_list[:per_dom])
        pairs = sampled[:num_pairs]

    print(f"Selected {len(pairs)} document pairs across domains: {sorted(list(set(p.domain for p in pairs)))}", flush=True)

    # 2. Instantiate Defense Matrix
    print("\n[Stage 2] Initializing 5-Architecture Defense Matrix (D1-D5)...", flush=True)
    defenses = load_defense_matrix(threshold=config.detection_threshold)
    for name, defense in defenses.items():
        print(f"  - {name:18}: {defense.__class__.__name__} (tau={defense.threshold})", flush=True)

    # 3. Instantiate Fidelity Judge
    print("\n[Stage 3] Initializing Independent Semantic Fidelity Judge...", flush=True)
    fidelity_judge = IndependentFidelityJudge(model_name=config.nli_model, threshold=config.fidelity_threshold)
    print(f"  - Fidelity Gate: threshold theta_fid={fidelity_judge.threshold} (Neural NLI: {fidelity_judge._using_nli})", flush=True)

    # 4. Instantiate LLM Generator & Attacks
    print("\n[Stage 4] Initializing Attack Engines...", flush=True)
    generator = LLMParaphraseGenerator(temperature=config.generation_temperature, top_p=config.generation_nucleus_p)
    
    tier1 = Tier1StaticAttack(budget=config.max_query_budget, fidelity_threshold=config.fidelity_threshold, generator=generator)
    tier2 = Tier2SaliencyAttack(
        budget=config.max_query_budget,
        fidelity_threshold=config.fidelity_threshold,
        top_k_spans=config.top_k_spans,
        candidate_pool_size=config.candidate_pool_size,
        generator=generator
    )

    # 5. Execute Attacks
    print("\n[Stage 5] Executing Paraphrase Attack Evaluations across Defense Matrix...", flush=True)
    all_results = []
    tier2_results_by_defense = {d_name: [] for d_name in defenses.keys()}
    
    total_evals = len(pairs) * len(defenses) * 3  # Tier1, Tier2, Corpus Adversarial
    completed = 0
    start_time = time.time()

    for p_idx, pair in enumerate(pairs, 1):
        print(f"\n--- Document Pair [{p_idx:03d}/{len(pairs):03d}]: [{pair.domain.upper()}] ID={pair.pair_id} | Topic: {pair.topic} ---", flush=True)
        for d_name, defense in defenses.items():
            # A. Tier 1 Static
            defense.reset_query_count()
            res_t1 = tier1.execute(
                source_text=pair.source_text,
                suspect_text=pair.suspect_text,
                defense=defense,
                fidelity_judge=fidelity_judge,
                pair_id=pair.pair_id,
                domain=pair.domain
            )
            all_results.append(res_t1)
            completed += 1
            st1 = "SUCCESS" if res_t1.is_fper else ("EVADED-LOW-FID" if res_t1.is_evasive else "CAUGHT")
            print(f"  [{completed:03d}/{total_evals:03d}] {d_name:18} | Tier1_Static       | s0={res_t1.initial_score:.3f} -> s'={res_t1.final_score:.3f} | Fid={res_t1.fidelity_score:.3f} | Q={res_t1.queries_consumed:2d} | {st1}", flush=True)

            # B. Tier 2 Saliency-Guided LOO
            defense.reset_query_count()
            res_t2 = tier2.execute(
                source_text=pair.source_text,
                suspect_text=pair.suspect_text,
                defense=defense,
                fidelity_judge=fidelity_judge,
                pair_id=pair.pair_id,
                domain=pair.domain
            )
            all_results.append(res_t2)
            tier2_results_by_defense[d_name].append(res_t2)
            completed += 1
            st2 = "SUCCESS" if res_t2.is_fper else ("EVADED-LOW-FID" if res_t2.is_evasive else "CAUGHT")
            print(f"  [{completed:03d}/{total_evals:03d}] {d_name:18} | Tier2_Saliency     | s0={res_t2.initial_score:.3f} -> s'={res_t2.final_score:.3f} | Fid={res_t2.fidelity_score:.3f} | Q={res_t2.queries_consumed:2d} | {st2}", flush=True)

            # C. Corpus Pre-populated Adversarial / Claude Paraphrase (if present)
            adv_text = pair.paraphrases.get("adversarial_paraphrase") or pair.paraphrases.get("claude_populated_paraphrase")
            if adv_text:
                strat_name = "Claude_Populated" if pair.paraphrases.get("claude_populated_paraphrase") else "Corpus_Adversarial"
                res_adv = evaluate_static_paraphrase_candidate(
                    pair_id=pair.pair_id,
                    domain=pair.domain,
                    source_text=pair.source_text,
                    suspect_text=pair.suspect_text,
                    candidate_text=adv_text,
                    strategy_name=strat_name,
                    defense=defense,
                    fidelity_judge=fidelity_judge,
                    budget=config.max_query_budget
                )
                all_results.append(res_adv)
                completed += 1
                st_adv = "SUCCESS" if res_adv.is_fper else ("EVADED-LOW-FID" if res_adv.is_evasive else "CAUGHT")
                print(f"  [{completed:03d}/{total_evals:03d}] {d_name:18} | {strat_name:18} | s0={res_adv.initial_score:.3f} -> s'={res_adv.final_score:.3f} | Fid={res_adv.fidelity_score:.3f} | Q={res_adv.queries_consumed:2d} | {st_adv}", flush=True)

    elapsed = time.time() - start_time
    print(f"\nCompleted {completed} evaluations in {elapsed:.2f}s ({elapsed/max(1, completed):.2f}s per eval).", flush=True)

    # 6. Compute Cross-Architecture Transferability Matrix
    print("\n[Stage 6] Computing 5x5 Cross-Architecture Transferability Matrix (T ∈ R^{5×5})...", flush=True)
    df_transfer = compute_transferability_matrix(
        optimized_results_by_defense=tier2_results_by_defense,
        defense_matrix=defenses,
        fidelity_judge=fidelity_judge
    )
    print("\nCross-Architecture Adversarial Transferability Matrix (T_{i,j} = FPER(P_{D_i}(x̃) -> D_j)):")
    print(df_transfer.round(3).to_string())

    # 7. Aggregate Metrics
    print("\n[Stage 7] Aggregating Performance Metrics...", flush=True)
    global_metrics = compute_full_metrics_summary(all_results)
    
    metrics_by_defense = {}
    for d_name in defenses.keys():
        d_res = [r for r in all_results if r.defense_name == d_name]
        metrics_by_defense[d_name] = compute_full_metrics_summary(d_res)

    metrics_by_tier = {}
    unique_tiers = set(r.attack_tier for r in all_results)
    for t_name in unique_tiers:
        t_res = [r for r in all_results if r.attack_tier == t_name]
        metrics_by_tier[t_name] = compute_full_metrics_summary(t_res)

    metrics_by_domain = {}
    for dom in set(p.domain for p in pairs):
        dom_res = [r for r in all_results if r.domain == dom]
        metrics_by_domain[dom] = compute_full_metrics_summary(dom_res)

    full_summary = {
        "global_metrics": global_metrics,
        "metrics_by_defense": metrics_by_defense,
        "metrics_by_tier": metrics_by_tier,
        "metrics_by_domain": metrics_by_domain,
        "transferability_matrix": df_transfer.to_dict()
    }

    # 8. Export Resulting Dataset and Visualizations
    print("\n[Stage 8] Exporting Datasets, Matrix & Visualization Plots...", flush=True)
    reporter = ExperimentReporter(output_dir=config.output_dir)
    
    dataset_paths = reporter.export_dataset(all_results)
    transfer_path = reporter.save_transferability_matrix(df_transfer)
    
    summary_path = os.path.join(config.output_dir, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)

    plot_paths = reporter.generate_plots(all_results, df_transfer)

    print("\n" + "=" * 85)
    print("EVALUATION & DATASET EXPORT COMPLETE")
    print("=" * 85)
    print(f"Resulting Dataset CSV:        {dataset_paths['csv_path']}")
    print(f"Resulting Dataset JSON:       {dataset_paths['json_path']}")
    print(f"Full Trajectories JSON:       {dataset_paths['full_json_path']}")
    print(f"Transferability Matrix CSV:   {transfer_path}")
    print(f"Summary Metrics JSON:         {summary_path}")
    for p_name, p_path in plot_paths.items():
        print(f"Generated Plot ({p_name}): {p_path}")
    print("=" * 85)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Adversarial Paraphrase Attack & Defense Benchmark")
    parser.add_argument("--num-pairs", type=int, default=15, help="Number of document pairs to evaluate (default: 15 for fast benchmarking, use 120 for full corpus)")
    parser.add_argument("--domain", type=str, default=None, help="Filter by specific domain (academic, padben, legal, scidocs, news)")
    args = parser.parse_args()
    
    run_evaluation_pipeline(num_pairs=args.num_pairs, domain_filter=args.domain)

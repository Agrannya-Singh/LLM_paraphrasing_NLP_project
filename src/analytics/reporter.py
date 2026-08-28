"""
Reporter & Visualizer.
Exports resulting dataset to CSV/JSON and generates visualization plots for empirical analysis.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from ..attacks.base import AttackResult

class ExperimentReporter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def export_dataset(self, results: List[AttackResult], base_filename: str = "dataset_evasion_results") -> Dict[str, str]:
        """
        Exports flat structured dataset for downstream analysis.
        """
        records = []
        for r in results:
            rec = {
                "pair_id": r.pair_id,
                "domain": r.domain,
                "defense_name": r.defense_name,
                "attack_tier": r.attack_tier,
                "initial_score": round(r.initial_score, 4),
                "final_score": round(r.final_score, 4),
                "score_drop": round(r.initial_score - r.final_score, 4),
                "threshold": r.threshold,
                "fidelity_score": round(r.fidelity_score, 4),
                "fidelity_threshold": r.fidelity_threshold,
                "is_evasive": r.is_evasive,
                "passes_fidelity": r.passes_fidelity,
                "is_fper": r.is_fper,
                "queries_consumed": r.queries_consumed,
                "source_text": r.source_text,
                "initial_suspect_text": r.initial_suspect_text,
                "final_paraphrase_text": r.final_paraphrase_text,
                "num_trajectory_steps": len(r.trajectory)
            }
            records.append(rec)
            
        df = pd.DataFrame(records)
        
        csv_path = os.path.join(self.output_dir, f"{base_filename}.csv")
        json_path = os.path.join(self.output_dir, f"{base_filename}.json")
        full_json_path = os.path.join(self.output_dir, f"{base_filename}_full_trajectories.json")
        
        df.to_csv(csv_path, index=False, encoding="utf-8")
        df.to_json(json_path, orient="records", indent=2)
        
        # Save complete object with step trajectories
        with open(full_json_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
            
        return {
            "csv_path": csv_path,
            "json_path": json_path,
            "full_json_path": full_json_path
        }

    def save_transferability_matrix(self, df_transfer: pd.DataFrame, filename: str = "transferability_matrix.csv") -> str:
        csv_path = os.path.join(self.output_dir, filename)
        df_transfer.to_csv(csv_path, encoding="utf-8")
        return csv_path

    def generate_plots(self,
                       results: List[AttackResult],
                       df_transfer: pd.DataFrame) -> Dict[str, str]:
        """
        Generates publication-quality charts: Transferability Heatmap, Attack Tier Evasion Rates, Domain Resilience.
        """
        sns.set_theme(style="whitegrid", palette="muted")
        plot_paths = {}
        
        # 1. Transferability Matrix Heatmap
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(df_transfer, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={'label': 'FPER (Transfer Evasion Rate)'}, vmin=0, vmax=1.0)
        plt.title("Cross-Architecture Adversarial Transferability Matrix (T ∈ R^{5×5})", fontsize=12, pad=12, fontweight='bold')
        plt.xlabel("Target Defense Oracle (D_j)", fontsize=10, labelpad=8)
        plt.ylabel("Source Optimization Defense (D_i)", fontsize=10, labelpad=8)
        plt.tight_layout()
        
        heatmap_path = os.path.join(self.output_dir, "plot_transferability_heatmap.png")
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        plot_paths["transferability_heatmap"] = heatmap_path

        # 2. Attack Tier Evasion & FPER Comparison
        df_res = pd.DataFrame([r.to_dict() for r in results])
        if not df_res.empty:
            plt.figure(figsize=(9, 5))
            tier_summary = df_res.groupby(["defense_name", "attack_tier"])[["is_evasive", "is_fper"]].mean().reset_index()
            tier_summary_melted = tier_summary.melt(id_vars=["defense_name", "attack_tier"], value_vars=["is_evasive", "is_fper"], var_name="Metric", value_name="Rate")
            tier_summary_melted["Metric"] = tier_summary_melted["Metric"].map({"is_evasive": "Evasion Rate (ER)", "is_fper": "Fidelity-Preserved ER (FPER)"})
            
            sns.barplot(data=tier_summary_melted, x="defense_name", y="Rate", hue="attack_tier", errorbar=None)
            plt.title("Defense Architecture Resistance Across Attack Tiers", fontsize=12, pad=12, fontweight='bold')
            plt.xlabel("Defense Architecture", fontsize=10)
            plt.ylabel("Success Rate", fontsize=10)
            plt.ylim(0, 1.05)
            plt.legend(title="Attack Strategy", bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()
            
            tier_plot_path = os.path.join(self.output_dir, "plot_attack_tiers_comparison.png")
            plt.savefig(tier_plot_path, dpi=300)
            plt.close()
            plot_paths["tier_comparison"] = tier_plot_path

            # 3. Query Complexity across Defenses
            plt.figure(figsize=(8, 5))
            t2_results = df_res[df_res["attack_tier"] == "Tier2_Saliency"]
            if not t2_results.empty:
                sns.boxplot(data=t2_results, x="defense_name", y="queries_consumed", palette="Blues")
                plt.title("Mean Query Complexity (MQC) Under Saliency-Guided Attack (Tier 2)", fontsize=12, pad=12, fontweight='bold')
                plt.xlabel("Defense Architecture", fontsize=10)
                plt.ylabel("Oracle Queries Consumed (B=50)", fontsize=10)
                plt.ylim(0, 55)
                plt.axhline(50, color='red', linestyle='--', alpha=0.7, label='Budget Limit B=50')
                plt.legend(loc="upper right")
                plt.tight_layout()
                
                query_plot_path = os.path.join(self.output_dir, "plot_query_complexity.png")
                plt.savefig(query_plot_path, dpi=300)
                plt.close()
                plot_paths["query_complexity"] = query_plot_path

        return plot_paths

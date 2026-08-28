import pandas as pd
import numpy as np

df = pd.read_csv("results/dataset_evasion_results.csv")

print("=== TABLE: MAIN RESULTS ===")
for d in ["D1_SBERT", "D2_SimCSE", "D3_BMX_Hybrid", "D4_ColBERT_MultiVector", "D5_Longformer"]:
    for t in ["Tier1_Static", "Tier2_Saliency", "Corpus_Adversarial"]:
        sub = df[(df["defense_name"] == d) & (df["attack_tier"] == t)]
        if len(sub) == 0:
            continue
        s0 = sub["initial_score"].mean()
        s_prime = sub["final_score"].mean()
        fid = sub["fidelity_score"].mean()
        er = sub["is_evasive"].mean() * 100
        fper = sub["is_fper"].mean() * 100
        mqc = sub["queries_consumed"].mean()
        
        # Simple bootstrap for 95% CI of FPER
        fpers = sub["is_fper"].values
        boot_fper = [np.mean(np.random.choice(fpers, len(fpers), replace=True)) * 100 for _ in range(1000)]
        fper_lb, fper_ub = np.percentile(boot_fper, 2.5), np.percentile(boot_fper, 97.5)
        
        print(f"{d} | {t} | s0={s0:.3f} | s'={s_prime:.3f} | F={fid:.3f} | ER={er:.1f}% | FPER={fper:.1f}% [{fper_lb:.1f}, {fper_ub:.1f}] | MQC={mqc:.1f}")


import json
import pandas as pd

with open("results/metrics_summary.json", "r", encoding="utf-8") as f:
    s = json.load(f)

print("=== GLOBAL METRICS ===")
for k, v in s["global_metrics"].items():
    print(f"  {k}: {v}")

print("\n=== DEFENSE BREAKDOWN ===")
for d, v in s["metrics_by_defense"].items():
    er = v.get("evasion_rate_er", 0)
    fper = v.get("fidelity_preserved_evasion_rate_fper", 0)
    mqc = v.get("mean_query_complexity_mqc", 0)
    sdrop = v.get("average_score_drop", 0)
    print(f"  {d:15s}: ER={er*100:.1f}%, FPER={fper*100:.1f}%, MQC={mqc:.1f}, AvgScoreDrop={sdrop:.4f}")

print("\n=== TIER BREAKDOWN ===")
for t, v in s["metrics_by_tier"].items():
    er = v.get("evasion_rate_er", 0)
    fper = v.get("fidelity_preserved_evasion_rate_fper", 0)
    mqc = v.get("mean_query_complexity_mqc", 0)
    print(f"  {t:20s}: ER={er*100:.1f}%, FPER={fper*100:.1f}%, MQC={mqc:.1f}")

print("\n=== DOMAIN BREAKDOWN ===")
for dom, v in s["metrics_by_domain"].items():
    er = v.get("evasion_rate_er", 0)
    fper = v.get("fidelity_preserved_evasion_rate_fper", 0)
    mqc = v.get("mean_query_complexity_mqc", 0)
    fid = v.get("average_fidelity_score", 0)
    print(f"  {dom:15s}: ER={er*100:.1f}%, FPER={fper*100:.1f}%, MQC={mqc:.1f}, Fidelity={fid:.3f}")

print("\n=== RAW TRANSFERABILITY MATRIX ===")
df_raw = pd.DataFrame(s["transferability_matrix_raw"])
print(df_raw.round(3))

print("\n=== NORMALIZED DELTA MATRIX ===")
df_delta = pd.DataFrame(s["transferability_matrix_normalized_delta"])
print(df_delta.round(3))

print("\n=== BMX ALPHA SENSITIVITY SWEEP ===")
for a, res in s["bmx_alpha_sensitivity_sweep"].items():
    print(f"  {a:10s}: MeanScore={res['mean_score']:.4f}, EvasionRate={res['evasion_rate']*100:.1f}%")

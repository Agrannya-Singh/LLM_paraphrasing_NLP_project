import pandas as pd

df = pd.read_csv("results/dataset_evasion_results.csv")

print("=== COUNTS BY DEFENSE x OUTCOME ===")
for d in df["defense_name"].unique():
    sub = df[df["defense_name"]==d]
    total = len(sub)
    evaded = int(sub["is_evasive"].sum())
    fper = int(sub["is_fper"].sum())
    avg_q = sub["queries_consumed"].mean()
    avg_fid = sub["fidelity_score"].mean()
    avg_sdrop = sub["score_drop"].mean()
    avg_init = sub["initial_score"].mean()
    below_tau = int((sub["initial_score"] < 0.75).sum())
    print(f"  {d:25s}: N={total}, Evaded={evaded}({evaded/total*100:.1f}%), FPER={fper}({fper/total*100:.1f}%), MQC={avg_q:.1f}, Fid={avg_fid:.3f}, ScoreDrop={avg_sdrop:.4f}")
    print(f"    AvgInitScore={avg_init:.4f}, PairsAlreadyBelowTau={below_tau}/{total} ({below_tau/total*100:.1f}%)")

print()
print("=== TIER BREAKDOWN ===")
for t in df["attack_tier"].unique():
    sub = df[df["attack_tier"]==t]
    total = len(sub)
    evaded = int(sub["is_evasive"].sum())
    fper = int(sub["is_fper"].sum())
    avg_q = sub["queries_consumed"].mean()
    print(f"  {t:25s}: N={total}, Evaded={evaded}({evaded/total*100:.1f}%), FPER={fper}({fper/total*100:.1f}%), MQC={avg_q:.1f}")

print()
print("=== FIDELITY-GATE FAILURES (Evaded but Failed Fidelity) ===")
evaded_no_fid = df[(df["is_evasive"]==True) & (df["passes_fidelity"]==False)]
print(f"  Total EVADED-LOW-FID: {len(evaded_no_fid)}/{len(df)} ({len(evaded_no_fid)/len(df)*100:.1f}%)")
for d in evaded_no_fid["defense_name"].unique():
    count = len(evaded_no_fid[evaded_no_fid["defense_name"]==d])
    print(f"    {d}: {count}")

print()
print("=== SCORE DYNAMICS (SimCSE vs BMX) ===")
for d in ["D2_SimCSE", "D3_BMX_Hybrid"]:
    sub = df[df["defense_name"]==d]
    print(f"  {d}: InitRange=[{sub['initial_score'].min():.3f}, {sub['initial_score'].max():.3f}], FinalRange=[{sub['final_score'].min():.3f}, {sub['final_score'].max():.3f}]")
    print(f"    AvgScoreDrop={sub['score_drop'].mean():.4f}, MaxScoreDrop={sub['score_drop'].max():.4f}")

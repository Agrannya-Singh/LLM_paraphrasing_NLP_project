# Retrieval and Similarity-Based Plagiarism Defenses Under Adaptive, Detector-Aware Paraphrase Attacks

An empirical framework implementing the benchmark and evaluation methodology presented in the paper *"Retrieval and Similarity-Based Plagiarism Defenses Under Adaptive, Detector-Aware Paraphrase Attacks"*.

---

## Architecture Overview

```
Stage 1: Corpus & Parsing
  ├── 5 Text Domains: Academic (PlagBench), Obfuscated (PADBen), Legal, SciDocs, News (SemEval-2022)
  ├── Pair Assembler (x, x̃)
  └── AST Syntactic Segmenter (M non-overlapping spans W)
          │
Stage 2: Defense Oracles (D1–D5)
  ├── D1: Sentence-BERT Bi-Encoder (all-MiniLM-L6-v2)
  ├── D2: SimCSE Bi-Encoder (sup-simcse-bert-base-uncased)
  ├── D3: BMX Hybrid Scorer (Dense Cosine + Entropy-Weighted Lexical Overlap)
  ├── D4: ColBERT Multi-Vector Late-Interaction Scorer (MaxSim Operator)
  └── D5: Longformer Sparse-Attention Document Scorer
          │
Stage 3: Adaptive Attack Engine
  ├── Tier 1: Static Paraphrase (Zero-feedback baseline)
  ├── Tier 2: Saliency-Guided Span Ablation (LOO I(w_i)) & Rejection Sampling
  └── Tier 3: Closed-Loop RL Reward Policy (PPO formulation)
          │
Stage 4: Fidelity Gate & Analytics
  ├── Architecturally Independent Gate (Non-BERT DeBERTa NLI Cross-Encoder)
  ├── Transferability Matrix T ∈ R^{5×5} (T_{i,j} = FPER(P_{D_i}(x̃) -> D_j))
  └── Quantitative Metrics (ER, FPER, MQC, Rank Correlation Decay Δρ)
```

---

## Mathematical Formulation

1. **Binary Detection Decision** (Eq. 1):
   $$\hat{y}(x, \tilde{x}) = \mathbb{I}[S(f_\theta(x), f_\theta(\tilde{x})) \ge \tau]$$

2. **Adversarial Optimization Objective** (Eq. 2):
   $$x' = \arg \min_{\hat{x} \in \mathcal{P}(\tilde{x})} S(x, \hat{x}) \quad \text{s.t.} \quad \mathcal{F}(x, \hat{x}) \ge \theta_{\text{fid}}, \quad Q(x') \le B$$

3. **BMX Hybrid Scorer** (Eq. 6):
   $$S_{D3}(x, \tilde{x}) = \alpha S_{\text{dense}}(u, v) + (1-\alpha) S_{\text{lex}}(x, \tilde{x})$$

4. **ColBERT MaxSim Late Interaction** (Eq. 7):
   $$S_{D4}(x, \tilde{x}) = \frac{1}{|x|} \sum_{i=1}^{|x|} \max_{j=1..|\tilde{x}|} \left( \frac{E_{x,i}^\top E_{\tilde{x},j}}{\|E_{x,i}\|_2 \|E_{\tilde{x},j}\|_2} \right)$$

5. **Leave-One-Out (LOO) Saliency Attribution** (Eq. 8):
   $$I(w_i) = |S(x, \tilde{x}) - S(x, \tilde{x} \setminus w_i)|$$

6. **Tier 3 RL Policy Reward** (Eq. 9):
   $$R(x, x') = -S_D(x, x') + \beta \mathcal{F}(x, x') - \gamma \max(0, \theta_{\text{fid}} - \mathcal{F}(x, x'))$$

7. **Fidelity-Preserved Evasion Rate (FPER)** (Eq. 10):
   $$\text{FPER} = \frac{1}{|D_{\text{plag}}|} \sum_{(x, \tilde{x}) \in D_{\text{plag}}} \mathbb{I}[S(x, x') < \tau \land \mathcal{F}(x, x') \ge \theta_{\text{fid}}]$$

8. **Adversarial Transferability Matrix** (Eq. 11):
   $$T_{i,j} = \text{FPER}(P_{D_i}(\tilde{x}) \to D_j)$$

---

## Directory Layout

```
LLM_paraphrasing_NLP_project/
├── .github/
│   └── workflows/
│       └── benchmarking.yml       # Continuous Benchmarking CI/CD Workflow
├── data/
│   └── benchmark_pairs.json       # Document pairs across 5 evaluation domains
├── src/
│   ├── config.py                  # Thresholds (τ=0.75, θ_fid=0.75), budget B=50
│   ├── corpus/
│   │   ├── loader.py              # PlagBench, PADBen, Legal, SciDocs, News
│   │   └── segmenter.py           # Syntactic span segmenter W = {w1..wM}
│   ├── defenses/
│   │   ├── base.py                # Base Defense Oracle with query counting
│   │   ├── sbert.py               # D1: Sentence-BERT (all-MiniLM-L6-v2)
│   │   ├── simcse.py              # D2: SimCSE (sup-simcse-bert-base-uncased)
│   │   ├── bmx.py                 # D3: BMX Hybrid (Dense + BM25 Lexical)
│   │   ├── colbert.py             # D4: ColBERT Multi-Vector MaxSim
│   │   └── longformer.py          # D5: Longformer Document Encoder
│   ├── fidelity/
│   │   └── judge.py               # Non-BERT DeBERTa NLI Cross-Encoder
│   ├── attacks/
│   │   ├── base.py                # Attack result & trajectory schema
│   │   ├── llm_generator.py       # Gemini 3.7 Flash Sampling Kernel (T=0.90, p=0.95)
│   │   ├── tier1_static.py        # Tier 1: Static Paraphrase
│   │   ├── tier2_saliency.py      # Tier 2: Saliency-Guided LOO & Rejection Sampling
│   │   └── tier3_rl.py            # Tier 3: Closed-Loop RL Reward Policy
│   └── analytics/
│       ├── metrics.py             # ER, FPER, MQC, Spearman rank decay (Δρ)
│       ├── transferability.py     # 5x5 Cross-Architecture Transferability Matrix
│       └── reporter.py            # Dataset exporter and publication visualizer
├── experiments/
│   └── run_pipeline.py            # Automated runner across all domains & defenses
├── tests/
│   └── test_framework.py          # Unit & integration test suite
├── results/
│   ├── dataset_evasion_results.csv
│   ├── dataset_evasion_results.json
│   ├── dataset_evasion_results_full_trajectories.json
│   ├── transferability_matrix.csv
│   ├── metrics_summary.json
│   └── *.png                      # Generated figures and heatmaps
├── requirements.txt
└── README.md
```

---

## Running Locally

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### 3. Run Benchmark Pipeline
```bash
python experiments/run_pipeline.py
```

---

## Continuous Benchmarking via GitHub Actions

This repository includes a turnkey GitHub Actions workflow (`.github/workflows/benchmarking.yml`) that:
1. Triggers automatically on every `push` and `pull_request` to `main`.
2. Runs scheduled weekly benchmarks on a cron schedule (`0 0 * * 0`).
3. Caches HuggingFace model weights (`~/.cache/huggingface`) across workflow runs.
4. Executes the full matrix evaluation and uploads `dataset_evasion_results.csv`, `transferability_matrix.csv`, and visualization heatmaps as workflow run artifacts.

### Setting up on GitHub:
1. Initialize git and push to your GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Adversarial paraphrase defense benchmark framework"
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
2. In your GitHub repository settings, go to **Settings > Secrets and variables > Actions** and add:
   - `GEMINI_API_KEY`: Your Gemini API key for LLM paraphraser sampling.

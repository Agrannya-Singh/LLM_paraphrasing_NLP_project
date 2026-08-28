# Git Policy & Project Directives

## 1. Commit Policy
- **Automatic Commits**: Create a structured git commit after each successfully completed prompt and milestone.
- **Commit Message Convention**:
  - `feat:` New features, defenses, corpus generators, or analytics modules.
  - `fix:` Bug fixes, threshold adjustments, or edge-case handling.
  - `test:` Unit and integration test additions or updates.
  - `ci:` GitHub Actions workflow configurations and runner optimizations.
  - `docs:` Documentation, walkthroughs, or README updates.

## 2. Zero External LLM Dependency Policy
- **Offline & Self-Contained**: The benchmarking pipeline operates 100% locally or on standard GitHub Actions runners with no external LLM API keys (zero Gemini/OpenAI API dependencies during CI runtime).
- **Static Corpus Benchmarking**: Document pairs (120 pairs across 5 domains) reside in `data/benchmark_pairs.json`.
- **Pre-populated Paraphrases**: Claude, GPT, or human-generated paraphrases are stored directly within the `"paraphrases"` dictionary of each benchmark item for reproducible, deterministic evaluation against the 5 defense oracles ($D_1$ to $D_5$) and the independent NLI fidelity gate.

## 3. Continuous Benchmarking CI/CD
- **Workflow**: `.github/workflows/benchmarking.yml`
- **Execution Matrix**: Python 3.10, 3.11, 3.12 on `ubuntu-latest`.
- **Artifacts**: Automatic publication of `dataset_evasion_results.csv`, `transferability_matrix.csv`, `metrics_summary.json`, and analytical heatmap figures on every push.

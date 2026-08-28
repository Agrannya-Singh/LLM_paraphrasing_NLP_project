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
- **Static Corpus Benchmarking**: Document pairs (250 pairs across 5 canonical benchmark domains) reside in `data/benchmark_pairs.json`.
- **Pre-populated Paraphrases**: Claude, GPT, or human-generated paraphrases are stored directly within the `"paraphrases"` dictionary of each benchmark item for reproducible, deterministic evaluation against the 5 defense oracles ($D_1$ to $D_5$) and the independent NLI fidelity gate.

## 3. Benchmarking CI/CD Policy
- **Workflows**: 
  - `.github/workflows/benchmarking.yml` (Adversarial Defense Benchmarking)
  - `.github/workflows/regression_sanity.yml` (Sanity Regression Test Suite & Verification)
  - `.github/workflows/compile_latex.yml` (LaTeX Paper PDF Build)
- **Trigger Policy**: Configured strictly for on-demand manual execution (`workflow_dispatch`). No actions run automatically on commits or pull requests to conserve GitHub Actions runner minutes.
- **Execution Matrix**: Python 3.10, 3.11, 3.12 on `ubuntu-latest`.
- **Artifacts**: Publication of `dataset_evasion_results.csv`, `transferability_matrix.csv`, `metrics_summary.json`, `Adversarial_Plagiarism_Benchmarking_Paper_PDF`, and analytical figures.

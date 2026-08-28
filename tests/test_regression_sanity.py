"""
Comprehensive Sanity & Regression Test Suite
Validates:
1. Benchmarking Dataset Integrity (250 pairs, schema, non-empty text, domain counts)
2. LaTeX Manuscript Structure (Sections, Equations, Tables, Algorithm, Bibliography >= 20)
3. Graph Generation & Figure Extraction to LaTeX
4. LaTeX Compilation Assets & Linkage
"""

import unittest
import json
import os
import re
import sys

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class TestSanityRegression(unittest.TestCase):
    def setUp(self):
        self.data_path = os.path.join(PROJECT_ROOT, "data", "benchmark_pairs.json")
        self.tex_path = os.path.join(PROJECT_ROOT, "paper", "main.tex")
        self.paper_dir = os.path.join(PROJECT_ROOT, "paper")
        self.fig_resistance = os.path.join(self.paper_dir, "defense_resistance.png")
        self.fig_complexity = os.path.join(self.paper_dir, "query_complexity.png")

    # -------------------------------------------------------------
    # 1. Benchmarking Dataset Sanity & Schema Validation
    # -------------------------------------------------------------
    def test_01_dataset_exists_and_valid_json(self):
        """Checks data/benchmark_pairs.json exists and is valid JSON."""
        self.assertTrue(os.path.exists(self.data_path), "benchmark_pairs.json does not exist")
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIsInstance(data, list, "Dataset root must be a JSON array")
        self.assertGreaterEqual(len(data), 250, "Dataset must contain at least 250 pairs")

    def test_02_dataset_domain_distribution(self):
        """Validates that each canonical domain contains at least 50 cross-sampled pairs."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        expected_domains = ["legal", "scidocs", "academic", "news", "padben"]
        domain_counts = {d: 0 for d in expected_domains}

        for item in data:
            d = item.get("domain")
            self.assertIn(d, expected_domains, f"Unexpected domain '{d}' in item {item.get('pair_id')}")
            domain_counts[d] += 1

        for domain, count in domain_counts.items():
            self.assertGreaterEqual(count, 50, f"Domain '{domain}' has {count} pairs (expected >= 50)")

    def test_03_dataset_item_schema_integrity(self):
        """Validates that every single pair item conforms strictly to the evaluation schema."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        required_keys = ["pair_id", "domain", "benchmark", "topic", "source_text", "suspect_text", "paraphrases", "metadata"]
        seen_ids = set()

        for idx, item in enumerate(data):
            for key in required_keys:
                self.assertIn(key, item, f"Missing '{key}' in item index {idx}")

            pair_id = item["pair_id"]
            self.assertNotIn(pair_id, seen_ids, f"Duplicate pair_id found: {pair_id}")
            seen_ids.add(pair_id)

            # Ensure texts are substantial and non-trivial
            self.assertGreater(len(item["source_text"].strip().split()), 5, f"Source text too short in {pair_id}")
            self.assertGreater(len(item["suspect_text"].strip().split()), 5, f"Suspect text too short in {pair_id}")
            
            # Ensure metadata contains access details
            self.assertIn("access_link", item["metadata"], f"Missing access_link in metadata of {pair_id}")

    # -------------------------------------------------------------
    # 2. LaTeX Research Paper Structural Regression Test
    # -------------------------------------------------------------
    def test_04_latex_file_exists_and_readable(self):
        """Checks paper/main.tex exists and is non-empty."""
        self.assertTrue(os.path.exists(self.tex_path), "paper/main.tex does not exist")
        with open(self.tex_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertGreater(len(content), 5000, "paper/main.tex is too short to be a complete manuscript")

    def test_05_latex_required_sections_present(self):
        """Validates all mandatory Phase 4 sections and IEEE format requirements."""
        with open(self.tex_path, "r", encoding="utf-8") as f:
            tex = f.read()

        self.assertIn("\\documentclass", tex)
        self.assertIn("{IEEEtran}", tex)
        self.assertIn("\\begin{abstract}", tex)
        self.assertIn("\\end{abstract}", tex)
        self.assertIn("\\begin{IEEEkeywords}", tex)

        required_sections = [
            "\\section{Introduction}",
            "\\section{Related Work}",
            "\\section{System Architecture \\& Methodology}",
            "\\section{Experimental Results \\& Discussion}",
            "\\section{Conclusion"
        ]
        for sec in required_sections:
            self.assertIn(sec, tex, f"Mandatory section '{sec}' missing from paper/main.tex")

    def test_06_latex_elements_architecture_algorithm_tables(self):
        """Checks presence of the Block Architecture figure, Algorithm 1 block, and all 4 tables."""
        with open(self.tex_path, "r", encoding="utf-8") as f:
            tex = f.read()

        # Architecture figure
        self.assertIn("\\label{fig:arch}", tex, "Architecture figure label fig:arch missing")
        # Algorithm block
        self.assertIn("\\begin{algorithm}", tex, "Algorithm block missing")
        self.assertIn("\\label{alg:saliency_attack}", tex, "Algorithm label alg:saliency_attack missing")
        
        # Tables 1 to 4
        self.assertIn("\\label{tab:main_results}", tex, "Table 1 (main_results) missing")
        self.assertIn("\\label{tab:domain_breakdown}", tex, "Table 2 (domain_breakdown) missing")
        self.assertIn("\\label{tab:transferability}", tex, "Table 3 (transferability) missing")
        self.assertIn("\\label{tab:baseline_comparison}", tex, "Table 4 (baseline_comparison) missing")

    def test_07_latex_bibliography_citations_count(self):
        """Ensures at least 20 academic references exist in the bibliography as per guidelines."""
        with open(self.tex_path, "r", encoding="utf-8") as f:
            tex = f.read()

        bibitems = re.findall(r"\\bibitem\{([^}]+)\}", tex)
        self.assertGreaterEqual(len(bibitems), 20, f"Found {len(bibitems)} references in bibliography (expected >= 20)")

        # Verify key canonical datasets are cited
        required_citations = ["oliveira2022legal", "mysore2022scientific", "goel2022wolfies", "zha2025padben", "lee2025plagbench"]
        for cite_key in required_citations:
            self.assertIn(cite_key, bibitems, f"Canonical citation '{cite_key}' missing from bibliography")

    # -------------------------------------------------------------
    # 3. Plotting & LaTeX Figure Extraction
    # -------------------------------------------------------------
    def test_08_graph_files_exist_and_non_empty(self):
        """Verifies that generated PNG plots exist and are non-zero byte images."""
        self.assertTrue(os.path.exists(self.fig_resistance), "paper/defense_resistance.png does not exist")
        self.assertTrue(os.path.exists(self.fig_complexity), "paper/query_complexity.png does not exist")
        
        self.assertGreater(os.path.getsize(self.fig_resistance), 1000, "defense_resistance.png is corrupted/empty")
        self.assertGreater(os.path.getsize(self.fig_complexity), 1000, "query_complexity.png is corrupted/empty")

    def test_09_graph_inclusion_in_latex(self):
        """Verifies that paper/main.tex properly references and embeds the generated PNG figures."""
        with open(self.tex_path, "r", encoding="utf-8") as f:
            tex = f.read()

        self.assertIn("defense_resistance.png", tex, "defense_resistance.png not referenced via \\includegraphics in main.tex")
        self.assertIn("query_complexity.png", tex, "query_complexity.png not referenced via \\includegraphics in main.tex")
        self.assertIn("\\label{fig:tier_comparison}", tex, "Figure label fig:tier_comparison missing")
        self.assertIn("\\label{fig:query_curve}", tex, "Figure label fig:query_curve missing")

if __name__ == "__main__":
    unittest.main()

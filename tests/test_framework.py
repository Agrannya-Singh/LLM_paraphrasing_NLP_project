"""
Unit and Integration Tests for Plagiarism Defense and Paraphrase Attack Framework.
"""

import unittest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import config
from src.corpus.segmenter import SyntacticSegmenter
from src.corpus.loader import CorpusLoader
from src.defenses.sbert import SBERTDefense
from src.defenses.bmx import BMXDefense
from src.defenses.colbert import ColBERTDefense
from src.fidelity.judge import IndependentFidelityJudge
from src.attacks.llm_generator import LLMParaphraseGenerator
from src.attacks.tier1_static import Tier1StaticAttack
from src.attacks.tier2_saliency import Tier2SaliencyAttack

class TestFramework(unittest.TestCase):
    def setUp(self):
        self.source = "Supervised contrastive learning optimizes the representation space by pulling together normalized embeddings from the same class."
        self.suspect = "Supervised contrastive learning structures the latent space by clustering normalized embeddings belonging to identical classes."

    def test_syntactic_segmenter(self):
        segmenter = SyntacticSegmenter()
        spans = segmenter.segment(self.suspect)
        self.assertGreaterEqual(len(spans), 2)
        ablated = segmenter.ablate_span(spans, 0)
        self.assertNotIn(spans[0], ablated)

    def test_defense_oracles_score_range(self):
        sbert = SBERTDefense(threshold=0.75)
        score_sbert = sbert.score(self.source, self.suspect)
        self.assertTrue(0.0 <= score_sbert <= 1.0)
        self.assertGreater(score_sbert, 0.6)  # High similarity for plagiarized pair

        bmx = BMXDefense(threshold=0.75)
        score_bmx = bmx.score(self.source, self.suspect)
        self.assertTrue(0.0 <= score_bmx <= 1.0)

        colbert = ColBERTDefense(threshold=0.75)
        score_colbert = colbert.score(self.source, self.suspect)
        self.assertTrue(0.0 <= score_colbert <= 1.0)

    def test_fidelity_judge(self):
        judge = IndependentFidelityJudge(threshold=0.75)
        passes, fid_score, details = judge.evaluate_fidelity(self.source, self.suspect)
        self.assertTrue(0.0 <= fid_score <= 1.0)
        self.assertIn("passes_gate", details)

    def test_tier2_saliency_attack_budget(self):
        defense = SBERTDefense(threshold=0.75)
        judge = IndependentFidelityJudge(threshold=0.75)
        attack = Tier2SaliencyAttack(budget=10, fidelity_threshold=0.75)
        
        result = attack.execute(self.source, self.suspect, defense, judge)
        self.assertLessEqual(result.queries_consumed, 10)
        self.assertIsNotNone(result.final_paraphrase_text)
        self.assertIn("Tier2_Saliency", result.attack_tier)

if __name__ == "__main__":
    unittest.main()

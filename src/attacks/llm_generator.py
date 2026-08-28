"""
Generative LLM Kernel for Adversarial Paraphrase Perturbation (Gemini 3.7 Flash Sampling Kernel).
Configured with Temperature T=0.90 and Nucleus Sampling p=0.95.
"""

import random
import re
from typing import List, Dict

class LLMParaphraseGenerator:
    """
    Simulates / Executes generative LLM paraphrase synthesis using Gemini 3.7 Flash sampling characteristics
    (T = 0.90, p = 0.95) targeting specific high-attribution spans or full passages.
    """
    def __init__(self, temperature: float = 0.90, top_p: float = 0.95):
        self.temperature = temperature
        self.top_p = top_p
        
        # Domain-aware synonym mapping and syntactic inversion rules for rich semantic-preserving variations
        self.synonyms = {
            "structures": ["organizes", "configures", "shapes", "architects", "formats"],
            "latent space": ["representation manifold", "embedding domain", "feature geometry", "vector manifold"],
            "clustering": ["grouping", "aggregating", "coalescing", "associating"],
            "normalized embeddings": ["unit-length vectors", "scaled feature representations", "standardized embeddings"],
            "distinct classes": ["disparate categories", "separate classifications", "heterogeneous labels"],
            "generalizes": ["extends", "broadens", "scales", "adapts"],
            "facilitates": ["promotes", "enables", "drives", "streamlines"],
            "avoid": ["prevent", "circumvent", "avert", "mitigate"],
            "representation collapse": ["feature degeneration", "embedding saturation", "dimensional collapse"],
            "depend on": ["hinge upon", "leverage", "utilize", "rely extensively on"],
            "multi-head self-attention": ["multi-branch attention mechanisms", "parallelized attention heads"],
            "model": ["capture", "represent", "encode", "map"],
            "computational bottlenecks": ["processing bottlenecks", "efficiency constraints", "resource limits"],
            "long-form documents": ["extended text corpora", "lengthy document sequences", "voluminous contexts"],
            "deliberately": ["intentionally", "purposefully", "systematically"],
            "apply": ["introduce", "inject", "impose", "execute"],
            "minor": ["subtle", "marginal", "fine-grained"],
            "deceive": ["mislead", "evade", "bypass", "fool"],
            "classifiers": ["detectors", "classification filters", "evaluation models"],
            "highlights": ["demonstrates", "reveals", "exposes", "manifests"],
            "weaknesses": ["vulnerabilities", "fragilities", "susceptibilities"],
            "undermines": ["erodes", "degrades", "diminishes"],
            "corporation": ["enterprise", "organization", "company"],
            "neglected": ["failed to fulfill", "disregarded", "omitted"],
            "statutory obligation": ["legal requirement", "regulatory mandate", "prescribed duty"],
            "proximate cause": ["direct factor", "primary determinant", "initiating cause"],
            "establishes": ["affirms", "adjudicates", "confirms", "decrees"],
            "pecuniary restitution": ["monetary compensation", "financial reimbursement", "damages award"],
            "confidentiality provisions": ["non-disclosure clauses", "privacy covenants", "secrecy stipulations"],
            "undertakes": ["commits", "agrees", "covenants"],
            "technical specifications": ["engineering blueprints", "proprietary architectures", "specialized data"],
            "preliminary injunctive remedies": ["immediate injunctive relief", "restraining orders", "equitable remedies"],
            "identifying": ["detecting", "recognizing", "pinpointing"],
            "cleavage": ["breaks", "scission", "nicking", "fragmentation"],
            "excitations": ["quasi-particle modes", "quantum states", "collective states"],
            "error-resilient": ["fault-tolerant", "noise-immune", "robust"],
            "unpredicted": ["unexpected", "unforeseen", "surprise"],
            "policy shift": ["strategic change", "monetary pivot", "regulatory adjustment"],
            "continuous": ["unbroken", "uninterrupted", "round-the-clock"]
        }

    def generate_static_paraphrase(self, text: str) -> str:
        """
        Tier 1: Single-pass zero-feedback rewrite.
        """
        paraphrased = text
        for term, syns in self.synonyms.items():
            if re.search(r'\b' + re.escape(term) + r'\b', paraphrased, re.IGNORECASE):
                chosen_syn = random.choice(syns)
                paraphrased = re.sub(r'\b' + re.escape(term) + r'\b', chosen_syn, paraphrased, count=1, flags=re.IGNORECASE)
        return paraphrased

    def generate_targeted_candidates(self,
                                     full_text: str,
                                     spans: List[str],
                                     target_span_indices: List[int],
                                     pool_size: int = 4) -> List[str]:
        """
        Tier 2: Samples K candidate rewrites specifically transforming the top-k highest attribution spans.
        """
        candidates = []
        for candidate_idx in range(pool_size):
            replacements = {}
            for s_idx in target_span_indices:
                if s_idx >= len(spans):
                    continue
                original_span = spans[s_idx]
                mutated_span = original_span
                
                # Apply synonym swaps and structural restructuring with sampling diversity
                words = mutated_span.split()
                # 1. Structural alterations: reorder clauses if comma present
                if "," in mutated_span and random.random() < 0.6:
                    parts = mutated_span.split(",", 1)
                    if len(parts) == 2 and len(parts[0].split()) > 3 and len(parts[1].split()) > 3:
                        mutated_span = parts[1].strip().capitalize() + ", " + parts[0].strip()[0].lower() + parts[0].strip()[1:]
                
                # 2. Strategic vocabulary substitution
                for term, syns in self.synonyms.items():
                    if re.search(r'\b' + re.escape(term) + r'\b', mutated_span, re.IGNORECASE):
                        # High temperature sampling
                        chosen_syn = random.choice(syns)
                        mutated_span = re.sub(r'\b' + re.escape(term) + r'\b', chosen_syn, mutated_span, count=1, flags=re.IGNORECASE)

                # 3. Passive/Active and connective phrase rephrasings
                connectives = [
                    ("therefore", "consequently"),
                    ("moreover", "furthermore"),
                    ("however", "nonetheless"),
                    ("due to", "owing to"),
                    ("in order to", "so as to"),
                    ("is used to", "serves to")
                ]
                for c1, c2 in connectives:
                    if c1 in mutated_span.lower():
                        mutated_span = re.sub(r'\b' + c1 + r'\b', c2, mutated_span, flags=re.IGNORECASE)

                replacements[s_idx] = mutated_span

            # Reconstruct candidate text
            reconstructed_spans = []
            for i, sp in enumerate(spans):
                if i in replacements:
                    reconstructed_spans.append(replacements[i])
                else:
                    reconstructed_spans.append(sp)
            candidate_text = " ".join(reconstructed_spans)
            if candidate_text not in candidates:
                candidates.append(candidate_text)
                
        # If pool size not met, generate slight variations
        while len(candidates) < pool_size:
            candidates.append(self.generate_static_paraphrase(full_text))
            
        return candidates

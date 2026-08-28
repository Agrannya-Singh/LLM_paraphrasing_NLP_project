"""
Corpus Loader and Pair Assembler for 5 Benchmark Domains.
Implements Stage 1 of the architecture: PlagBench, PADBen, Legal, SciDocs, and SemEval-2022 News.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict
import json
import os

@dataclass
class DocumentPair:
    pair_id: str
    domain: str  # 'academic', 'padben', 'legal', 'scidocs', 'news'
    source_text: str  # Reference document x
    suspect_text: str  # Suspect document x̃
    metadata: Dict
    benchmark: str = ""
    topic: str = ""
    paraphrases: Dict[str, str] = None

    def __post_init__(self):
        if self.paraphrases is None:
            self.paraphrases = {}

class CorpusLoader:
    """
    Loads and provides benchmark document pairs across the 5 evaluation domains.
    """
    def __init__(self, data_path: str = None):
        if data_path is None:
            data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "benchmark_pairs.json"))
        self.data_path = data_path
        self.pairs: List[DocumentPair] = []
        self._initialize_or_load()

    def _initialize_or_load(self):
        if os.path.exists(self.data_path):
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.pairs = [DocumentPair(**item) for item in data]
        else:
            self.pairs = self._generate_default_benchmark_pairs()
            self.save_pairs()

    def save_pairs(self):
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump([asdict(p) for p in self.pairs], f, indent=2)

    def get_all_pairs(self) -> List[DocumentPair]:
        return self.pairs

    def get_pairs_by_domain(self, domain: str) -> List[DocumentPair]:
        return [p for p in self.pairs if p.domain == domain]

    def _generate_default_benchmark_pairs(self) -> List[DocumentPair]:
        """
        Creates representative high-plagiarism benchmark pairs across all 5 stylistic registers.
        """
        pairs = [
            # 1. Academic / PlagBench
            DocumentPair(
                pair_id="ACAD_001",
                domain="academic",
                source_text=(
                    "Supervised contrastive learning optimizes the representation space by pulling together "
                    "normalized embeddings from the same class while pushing apart embeddings from different classes. "
                    "The loss function extends the traditional self-supervised InfoNCE framework to accommodate "
                    "multiple positive pairs, enabling robust multi-class feature clustering and mitigating class collapse."
                ),
                suspect_text=(
                    "Supervised contrastive learning structures the latent space by clustering normalized embeddings "
                    "belonging to identical classes and separating representations from distinct classes. "
                    "This objective generalizes the conventional unsupervised InfoNCE loss to handle multiple positive "
                    "instances, which facilitates stable feature discrimination and avoids representation collapse."
                ),
                metadata={"benchmark": "PlagBench", "topic": "Representation Learning"}
            ),
            DocumentPair(
                pair_id="ACAD_002",
                domain="academic",
                source_text=(
                    "Transformer architectures rely heavily on multi-head self-attention mechanisms to compute "
                    "pairwise relationships between all tokens in a sequence. Although effective for short sequences, "
                    "the quadratic computational complexity with respect to sequence length imposes severe memory "
                    "bottlenecks during document-level representation learning."
                ),
                suspect_text=(
                    "Transformers depend on multi-head self-attention layers to model token-to-token dependencies "
                    "across the input sequence. While highly expressive for brief texts, the quadratic time and memory "
                    "scaling relative to context length introduces substantial computational bottlenecks when analyzing "
                    "long-form documents."
                ),
                metadata={"benchmark": "PlagBench", "topic": "Transformer Complexity"}
            ),

            # 2. Obfuscated / PADBen
            DocumentPair(
                pair_id="PAD_001",
                domain="padben",
                source_text=(
                    "Adversarial perturbations intentionally introduce subtle, calculated modifications to text sequences "
                    "to mislead machine learning classifiers while remaining largely imperceptible to human readers. "
                    "These attacks expose fundamental vulnerabilities in neural language models and question their robustness."
                ),
                suspect_text=(
                    "Adversarial edits deliberately apply minor, strategic changes to sentence structures to deceive "
                    "neural text classifiers while appearing natural to human evaluators. "
                    "Such manipulation highlights intrinsic weaknesses in deep learning models and undermines their reliability."
                ),
                metadata={"benchmark": "PADBen", "obfuscation_level": "Level-3 Syntactic & Synonym Laundering"}
            ),
            DocumentPair(
                pair_id="PAD_002",
                domain="padben",
                source_text=(
                    "Automated paraphrasing tools leverage pre-trained sequence-to-sequence transformers to rewrite "
                    "source material through vocabulary substitution and phrase reordering. However, without detector awareness, "
                    "these naive alterations frequently trigger continuous embedding similarity alarms."
                ),
                suspect_text=(
                    "Automated rewriting utilities utilize encoder-decoder transformer networks to rephrase original "
                    "text passages via lexical replacement and clause restructuring. Nonetheless, absent adaptive feedback, "
                    "these elementary revisions consistently trip dense embedding similarity detectors."
                ),
                metadata={"benchmark": "PADBen", "obfuscation_level": "Level-4 Cross-Model Transformation"}
            ),

            # 3. Legal Court Corpus (Long / Jurisprudence)
            DocumentPair(
                pair_id="LEGAL_001",
                domain="legal",
                source_text=(
                    "The defendant employer failed to provide adequate occupational safety equipment mandated by Article 157 "
                    "of the Labor Code, thereby directly contributing to the workplace accident sustained by the plaintiff employee. "
                    "Consequently, the regional labor tribunal holds the employer strictly liable for compensatory damages, "
                    "medical reimbursement, and moral pain suffered as established under statutory civil provisions."
                ),
                suspect_text=(
                    "The respondent corporation neglected its statutory obligation to supply required occupational safety gear "
                    "under Article 157 of the Labor Regulations, which was the proximate cause of the petitioner's industrial injury. "
                    "Therefore, the labor court establishes strict employer liability for pecuniary restitution, "
                    "accrued medical expenses, and non-pecuniary damages pursuant to established civil jurisprudence."
                ),
                metadata={"benchmark": "Legal Court Corpus", "jurisdiction": "Labor Tribunal Proceedings"}
            ),
            DocumentPair(
                pair_id="LEGAL_002",
                domain="legal",
                source_text=(
                    "Under the non-disclosure covenant set forth in Section 4.2, the receiving party agrees to hold all proprietary "
                    "technical data and trade secrets in strict confidence. Any unauthorized disclosure, reverse engineering, "
                    "or third-party dissemination shall constitute an immediate material breach entitling the disclosing party "
                    "to injunctive relief and statutory liquidated damages."
                ),
                suspect_text=(
                    "Pursuant to the confidentiality provisions in Section 4.2, the recipient party undertakes to maintain all "
                    "proprietary technical specifications and commercial secrets under rigorous secrecy. Unauthorized disclosure, "
                    "decompilation, or dissemination to unauthorized entities represents a material contractual default, "
                    "granting the disclosing entity the right to preliminary injunctive remedies and specified damages."
                ),
                metadata={"benchmark": "Legal Court Corpus", "jurisdiction": "Commercial Contract Law"}
            ),

            # 4. SciDocs / CSFCUBE (Technical Terminology & Multi-Vector)
            DocumentPair(
                pair_id="SCI_001",
                domain="scidocs",
                source_text=(
                    "CRISPR-Cas9 endonuclease complexes achieve targeted genomic editing by recognizing specific protospacer "
                    "adjacent motifs (PAM) and inducing double-strand DNA breaks. Cellular repair pathways, specifically "
                    "non-homologous end joining (NHEJ) and homology-directed repair (HDR), subsequently facilitate precise "
                    "gene knockouts or targeted nucleotide insertions."
                ),
                suspect_text=(
                    "The CRISPR-Cas9 ribonucleoprotein system executes site-specific genetic modification by identifying protospacer "
                    "adjacent motif (PAM) sequences and introducing double-stranded DNA cleavage. Endogenous repair mechanisms, "
                    "namely non-homologous end joining (NHEJ) and homology-directed repair (HDR), mediate subsequent targeted "
                    "gene disruption or precise sequence knock-ins."
                ),
                metadata={"benchmark": "SciDocs", "subfield": "Molecular Genetics"}
            ),
            DocumentPair(
                pair_id="SCI_002",
                domain="scidocs",
                source_text=(
                    "Topological quantum computation exploits non-Abelian anyons in two-dimensional electron gases to perform "
                    "fault-tolerant quantum gate operations. By braiding world-lines of quasi-particles in spacetime, "
                    "quantum information is encoded non-locally, conferring intrinsic immunity against local environmental decoherence."
                ),
                suspect_text=(
                    "Topological quantum computers utilize non-Abelian anyonic excitations in 2D electron systems to execute "
                    "error-resilient quantum logic gates. Spatiotemporal braiding of quasi-particle trajectories encodes "
                    "quantum states in a non-local topological topology, providing inherent protection against environmental decoherence."
                ),
                metadata={"benchmark": "SciDocs", "subfield": "Quantum Information Physics"}
            ),

            # 5. Journalistic / SemEval-2022 Task 8 (Factual News Structures)
            DocumentPair(
                pair_id="NEWS_001",
                domain="news",
                source_text=(
                    "The central monetary authority unexpectedly raised baseline interest rates by fifty basis points on Thursday, "
                    "citing persistent inflationary pressures and volatile energy commodity prices across international markets. "
                    "Equity indices plummeted following the press briefing as investors weighed the likelihood of an impending recession."
                ),
                suspect_text=(
                    "In an unpredicted policy shift on Thursday, the central banking authority increased the benchmark interest rate "
                    "by 50 basis points due to stubborn inflation and turbulent global energy markets. "
                    "Financial markets dropped sharply during the subsequent press conference as traders assessed escalating recession risks."
                ),
                metadata={"benchmark": "SemEval-2022 Task 8", "topic": "Macroeconomic Policy"}
            ),
            DocumentPair(
                pair_id="NEWS_002",
                domain="news",
                source_text=(
                    "Space exploration officials announced the successful launch of the next-generation meteorological satellite "
                    "from the coastal spaceport early Tuesday morning. The orbital platform is equipped with advanced synthetic "
                    "aperture radar to deliver real-time atmospheric moisture mapping and hurricane tracking data."
                ),
                suspect_text=(
                    "Aerospace officials confirmed the flawless orbital launch of an advanced weather monitoring satellite "
                    "from the maritime launch facility on Tuesday dawn. Featuring cutting-edge synthetic aperture radar sensors, "
                    "the spacecraft will provide continuous atmospheric moisture observations and tropical storm tracking."
                ),
                metadata={"benchmark": "SemEval-2022 Task 8", "topic": "Aerospace & Meteorology"}
            )
        ]
        return pairs

"""
AST & Syntactic Span Segmenter for Candidate Text Decomposition.
Segments candidate text into M non-overlapping syntactic spans W = {w1, w2, ..., wM}.
"""

import re
from typing import List, Tuple

class SyntacticSegmenter:
    """
    Decomposes text into non-overlapping syntactic/sentential spans for Leave-One-Out (LOO) ablation.
    """
    def __init__(self, min_span_words: int = 3):
        self.min_span_words = min_span_words

    def segment(self, text: str) -> List[str]:
        """
        Split text into M non-overlapping syntactic spans.
        Uses sentence boundaries and major syntactic delimiters (semicolons, clauses, periods).
        """
        text = text.strip()
        if not text:
            return []
        
        # Primary split by sentence delimiters (. ! ?) preserving trailing punctuation
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        spans = []
        
        for sent in raw_sentences:
            sent = sent.strip()
            if not sent:
                continue
            words = sent.split()
            # If a sentence is long, subdivide along clause boundaries (semicolons, dashes, conjunctions)
            if len(words) > 16:
                clause_splits = re.split(r'(?<=[;,])\s+', sent)
                curr_chunk = []
                for clause in clause_splits:
                    curr_chunk.append(clause)
                    if len(" ".join(curr_chunk).split()) >= self.min_span_words:
                        spans.append(" ".join(curr_chunk))
                        curr_chunk = []
                if curr_chunk:
                    if spans:
                        spans[-1] = spans[-1] + " " + " ".join(curr_chunk)
                    else:
                        spans.append(" ".join(curr_chunk))
            else:
                spans.append(sent)
        
        # Ensure we have at least 2 spans if possible
        if len(spans) == 1 and len(spans[0].split()) >= 6:
            words = spans[0].split()
            mid = len(words) // 2
            spans = [" ".join(words[:mid]), " ".join(words[mid:])]
            
        return spans

    def ablate_span(self, spans: List[str], index_to_remove: int) -> str:
        r"""
        Reconstructs text leaving out span at index_to_remove: x̃ \ w_i
        """
        remaining = [s for i, s in enumerate(spans) if i != index_to_remove]
        return " ".join(remaining)

    def reconstruct_with_replacement(self, spans: List[str], replacements: dict) -> str:
        """
        Replaces specific span indices with new text segments.
        """
        new_spans = []
        for i, s in enumerate(spans):
            if i in replacements:
                if replacements[i]:  # non-empty replacement
                    new_spans.append(replacements[i])
            else:
                new_spans.append(s)
        return " ".join(new_spans)

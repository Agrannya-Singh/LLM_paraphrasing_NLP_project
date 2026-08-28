"""
Corpus and Parsing Package
"""
from .segmenter import SyntacticSegmenter
from .loader import CorpusLoader, DocumentPair

__all__ = ["SyntacticSegmenter", "CorpusLoader", "DocumentPair"]

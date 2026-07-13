"""
Multi-Hop Question Decomposer

A rule-based library for decomposing compositional multi-hop questions
into hop-level sub-questions using spaCy.
"""

from .decomposer import MultiHopDecomposer, DecomposedHop, decompose

__version__ = "0.1.0"
__all__ = ["MultiHopDecomposer", "DecomposedHop", "decompose"]

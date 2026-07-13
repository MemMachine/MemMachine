"""
Multi-Hop Question Decomposer

A rule-based library for decomposing compositional multi-hop questions
into hop-level sub-questions using spaCy.

spaCy is an optional dependency (the ``multihop`` dependency-group). When it is
not installed, importing this package does not hard-fail: the symbols below
resolve to ``None`` and ``RaragQueryAgent`` falls back to LLM-based splitting.
"""

try:
    from .decomposer import MultiHopDecomposer, DecomposedHop, decompose
except ImportError:  # spaCy (optional, ``multihop`` group) is not installed.
    MultiHopDecomposer = None  # type: ignore[assignment,misc]
    DecomposedHop = None  # type: ignore[assignment,misc]
    decompose = None  # type: ignore[assignment,misc]

__version__ = "0.1.0"
__all__ = ["MultiHopDecomposer", "DecomposedHop", "decompose"]

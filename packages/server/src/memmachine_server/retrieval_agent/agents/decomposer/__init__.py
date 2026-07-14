"""
Multi-Hop Question Decomposer.

A rule-based library for decomposing compositional multi-hop questions
into hop-level sub-questions using spaCy.

spaCy is an optional dependency (the ``multihop`` dependency-group). When it is
not installed, importing this package does not hard-fail: the symbols below
resolve to ``None`` and ``RaragQueryAgent`` falls back to LLM-based splitting.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .decomposer import DecomposedHop, DecompositionResult, MultiHopDecomposer

# spaCy (optional, ``multihop`` group) is not installed by default. Pre-declare
# the symbols as ``None`` so importing this package never hard-fails; the
# import below rebinds them to the real class/function when spaCy is available.
MultiHopDecomposer: type[MultiHopDecomposer] | None = None
DecomposedHop: type[DecomposedHop] | None = None
decompose: Callable[..., DecompositionResult] | None = None

with contextlib.suppress(ImportError):
    from .decomposer import DecomposedHop, MultiHopDecomposer, decompose

__version__ = "0.1.0"
__all__ = ["DecomposedHop", "MultiHopDecomposer", "decompose"]

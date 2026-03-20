"""Segment store data models and interfaces."""

from .segment_linker import (
    DerivativeNotActiveError,
    SegmentLinker,
    SegmentLinkerPartition,
)

__all__ = [
    "DerivativeNotActiveError",
    "SegmentLinker",
    "SegmentLinkerPartition",
]

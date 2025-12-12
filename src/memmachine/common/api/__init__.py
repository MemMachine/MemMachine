"""Shared API definitions for MemMachine client and server."""

from enum import Enum


class MemoryType(Enum):
    """Memory type."""

    Semantic = "semantic"
    Episodic = "episodic"


__all__ = ["MemoryType"]

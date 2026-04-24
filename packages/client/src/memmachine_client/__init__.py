"""
MemMachine Client - A Python client library for MemMachine memory system.

This module provides a high-level interface for interacting with MemMachine's
episodic and profile memory systems.
"""

from .client import MemMachineClient
from .config import Config
from .format import (
    format_episodes,
    format_search_result,
    format_semantic_memories,
)
from .memory import Memory
from .project import Project

__all__ = [
    "Config",
    "MemMachineClient",
    "Memory",
    "Project",
    "format_episodes",
    "format_search_result",
    "format_semantic_memories",
]

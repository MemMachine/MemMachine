"""
MemMachine Client - A Python client library for MemMachine memory system.

This module provides a high-level interface for interacting with MemMachine's
episodic and profile memory systems.
"""

from .client import MemMachineClient
from .memory import Memory

__all__ = ["MemMachineClient", "Memory"]

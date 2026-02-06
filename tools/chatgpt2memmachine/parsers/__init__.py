"""
Chat history parsers package.

This package provides parsers for different chat history formats.
"""

from .base import BaseParser
from .locomo_parser import LocomoParser
from .openai_parser import OpenAIParser

__all__ = ["BaseParser", "LocomoParser", "OpenAIParser"]


def get_parser(source: str, verbose: bool = False):
    """
    Get the appropriate parser for the given source.

    Args:
        source: Source type ('openai' or 'locomo')
        verbose: Enable verbose logging

    Returns:
        Parser instance

    Raises:
        ValueError: If source type is unknown
    """
    source = source.lower()
    if source == "openai":
        return OpenAIParser(verbose=verbose)
    if source == "locomo":
        return LocomoParser(verbose=verbose)
    raise ValueError(f"Unknown input source: {source}")

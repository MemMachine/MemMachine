"""Public package exports and utilities for MemMachine."""

from memmachine.main.memmachine import MemMachine
from memmachine.rest_client import MemMachineClient, Memory


def setup_nltk() -> None:
    """Check for and download required NLTK data packages."""
    import logging

    import nltk

    logger = logging.getLogger(__name__)

    logger.info("Checking for required NLTK data...")
    packages = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, pkg_id in packages:
        try:
            nltk.data.find(path)
            logger.info("NLTK package '%s' is already installed.", pkg_id)
        except LookupError:
            logger.warning("NLTK package '%s' not found. Downloading...", pkg_id)
            nltk.download(pkg_id)
    logger.info("NLTK data setup is complete.")


__all__ = ["MemMachine", "MemMachineClient", "Memory", "setup_nltk"]

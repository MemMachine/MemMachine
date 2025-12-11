"""Public package exports and utilities for MemMachine."""

from memmachine.rest_client import MemMachineClient, Memory, Project

try:
    from memmachine.main.memmachine import MemMachine
except ImportError:
    # MemMachine is not available in client-only installations
    MemMachine = None  # type: ignore


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


# Only export MemMachine if it's available
if MemMachine is not None:
    __all__ = ["MemMachine", "MemMachineClient", "Memory", "Project", "setup_nltk"]
else:
    __all__ = ["MemMachineClient", "Memory", "Project", "setup_nltk"]

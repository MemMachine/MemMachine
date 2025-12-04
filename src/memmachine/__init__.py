"""Public package exports and utilities for MemMachine."""

try:
    from memmachine.rest_client import MemMachineClient, Memory, Project
except ImportError:
    # rest_client is not available in server-only installations
    MemMachineClient = None  # type: ignore
    Memory = None  # type: ignore
    Project = None  # type: ignore

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


# Only export available modules
exports = ["setup_nltk"]
if MemMachine is not None:
    exports.append("MemMachine")
if MemMachineClient is not None:
    exports.extend(["MemMachineClient", "Memory", "Project"])
__all__ = exports

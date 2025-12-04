"""Utility functions and constants for MemMachine installation scripts."""

from enum import Enum
from typing import Self

LINUX_JDK_TAR_NAME = "jdk-21_linux-x64_bin.tar.gz"
LINUX_JDK_URL = f"https://download.oracle.com/java/21/latest/{LINUX_JDK_TAR_NAME}"
WINDOWS_JDK_ZIP_NAME = "jdk-21_windows-x64_bin.zip"
WINDOWS_JDK_URL = f"https://download.oracle.com/java/21/latest/{WINDOWS_JDK_ZIP_NAME}"
NEO4J_DIR_NAME = "neo4j-community-2025.09.0"
WINDOWS_NEO4J_ZIP_NAME = f"{NEO4J_DIR_NAME}-windows.zip"
WINDOWS_NEO4J_URL = f"https://dist.neo4j.org/{WINDOWS_NEO4J_ZIP_NAME}"
LINUX_NEO4J_TAR_NAME = f"{NEO4J_DIR_NAME}-unix.tar.gz"
LINUX_NEO4J_URL = f"https://dist.neo4j.org/{LINUX_NEO4J_TAR_NAME}"
JDK_DIR_NAME = "jdk-21.0.9"
NEO4J_WINDOWS_SERVICE_NAME = "neo4j"

NEO4J_GPG_KEY_URL = "https://debian.neo4j.com/neotechnology.gpg.key"

NEO4J_GPG_KEY_PATH_DEB = "/etc/apt/keyrings/neotechnology.gpg"
NEO4J_DEB_REPO_ENTRY = "deb [signed-by=/etc/apt/keyrings/neotechnology.gpg] https://debian.neo4j.com stable latest"
NEO4J_DEB_SOURCE_LIST_PATH = "/etc/apt/sources.list.d/neo4j.list"

NEO4J_YUM_REPO_FILE_PATH = "/etc/yum.repos.d/neo4j.repo"
NEO4J_YUM_REPO_ENTRY = """[neo4j]
name=Neo4j RPM Repository
baseurl=https://yum.neo4j.com/stable/latest
enabled=1
gpgcheck=1
EOF"""


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_BEDROCK_MODEL = "openai.gpt-oss-20b-1:0"
DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_BEDROCK_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
DEFAULT_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_EMBEDDING_DIMENSIONS = 768

DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_NEO4J_USERNAME = "neo4j"
DEFAULT_NEO4J_PASSWORD = "memmachine"

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class ScriptType(Enum):
    """Enumeration of supported script types."""

    BASH = "bash"
    POWERSHELL = "powershell"


class ModelProvider(Enum):
    """Enumeration of supported language model providers."""

    OPENAI = "openai"
    BEDROCK = "bedrock"
    OLLAMA = "ollama"

    @classmethod
    def parse(cls, raw: str) -> Self:
        """
        Parse user-provided input (case-insensitive) and map it to a ModelProvider.

        Falls back to OPENAI on invalid input.
        """
        if not raw:
            return cls.OPENAI

        raw = raw.strip().lower()

        mapping = {
            "openai": cls.OPENAI,
            "bedrock": cls.BEDROCK,
            "ollama": cls.OLLAMA,
        }

        return mapping.get(raw, cls.OPENAI)

"""Utility functions and constants for MemMachine installation scripts."""

from enum import Enum

LINUX_JDK_TAR_NAME = "jdk-21_linux-x64_bin.tar.gz"
LINUX_JDK_URL = "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.12.1%2B1/OpenJDK21U-jdk_x64_linux_hotspot_21.0.12.1_1.tar.gz"
LINUX_JDK_SHA256 = "ce79869e1307ed8ee1e2baa86a412b1eb5b75d10a01006d788a6f968bcfaee94"
WINDOWS_JDK_ZIP_NAME = "jdk-21_windows-x64_bin.zip"
WINDOWS_JDK_URL = "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.12.1%2B1/OpenJDK21U-jdk_x64_windows_hotspot_21.0.12.1_1.zip"
WINDOWS_JDK_SHA256 = "f9d6e191ab098c0d416e7d588a24420a8621cd2f4720dab2459b8b7b2d2d8b4e"
NEO4J_DIR_NAME = "neo4j-community-2025.09.0"
WINDOWS_NEO4J_ZIP_NAME = f"{NEO4J_DIR_NAME}-windows.zip"
WINDOWS_NEO4J_URL = f"https://dist.neo4j.org/{WINDOWS_NEO4J_ZIP_NAME}"
WINDOWS_NEO4J_SHA256 = (
    "849daf891a6aec4f40777fb4e26cd85c659ae19208bd4842d8dde3e7657f5692"
)
LINUX_NEO4J_TAR_NAME = f"{NEO4J_DIR_NAME}-unix.tar.gz"
LINUX_NEO4J_URL = f"https://dist.neo4j.org/{LINUX_NEO4J_TAR_NAME}"
LINUX_NEO4J_SHA256 = "997928463c687a96433720610c3091969fd4d5f76f8a12d09a6149bddf65927a"

MACOS_JDK_TAR_NAME_X64 = "jdk-21_macos-x64_bin.tar.gz"
MACOS_JDK_URL_X64 = "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.12.1%2B1/OpenJDK21U-jdk_x64_mac_hotspot_21.0.12.1_1.tar.gz"
MACOS_JDK_SHA256_X64 = (
    "44db0f08196daf19a47f90d13388b0c943b67663cb537f998fe29e836fa842ce"
)
MACOS_JDK_TAR_NAME_ARM64 = "jdk-21_macos-aarch64_bin.tar.gz"
MACOS_JDK_URL_ARM64 = "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.12.1%2B1/OpenJDK21U-jdk_aarch64_mac_hotspot_21.0.12.1_1.tar.gz"
MACOS_JDK_SHA256_ARM64 = (
    "3623232f33a9c3baadf304480b2535f9a3cba8a58d42ecbb438ba267315d9998"
)

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


DEFAULT_OPENAI_MODEL = "gpt-6-luna"
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
    def parse(cls, raw: str) -> "ModelProvider":
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

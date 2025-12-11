"""Logging configuration helpers."""

import logging
import os
import sys
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator

from memmachine.common.configuration.mixin_confs import YamlSerializableMixin

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Supported logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def to_log_level(level_str: str) -> LogLevel:
    """Parse a string into a `LogLevel` enum, raising on invalid input."""
    try:
        return LogLevel[level_str.upper()]
    except KeyError:
        raise ValueError(f"Invalid log level: {level_str}") from None


class LogConf(YamlSerializableMixin):
    """Configuration model for application logging."""

    level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level",
    )
    format: str = Field(
        default="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        description="Logging format string",
    )
    path: str = Field(
        default="MemMachine.log",
        description="Path to log file (if empty, logs only to stdout).",
    )

    @field_validator("level", mode="before")
    @staticmethod
    def validate_level(v: str | LogLevel) -> LogLevel:
        """Normalize level input into a LogLevel enum."""
        if isinstance(v, LogLevel):
            return v
        try:
            return to_log_level(str(v))
        except ValueError as e:
            raise ValueError(f"Invalid log level: {v}") from e

    @field_validator("format")
    @staticmethod
    def validate_format(v: str) -> str:
        """Ensure format string contains basic logging tokens."""
        # A minimal sanity check: must include %(levelname)s and %(message)s
        required_tokens = ["%(levelname)s", "%(message)s", "%(asctime)s"]
        for token in required_tokens:
            if token not in v:
                raise ValueError(f"log format must include {token}, got '{v}'")
        return v

    @field_validator("path")
    @staticmethod
    def validate_path(v: str | None) -> str | None:
        """Validate that the log path directory exists and is writable."""
        if v is None or v == "":
            return None
        path = Path(v).expanduser()
        dir_path = path.parent if path.is_absolute() else (Path.cwd() / path).parent
        if not dir_path.exists():
            raise ValueError(f"Log directory does not exist: {dir_path}")
        if not os.access(dir_path, os.W_OK):
            raise ValueError(f"Log directory is not writable: {dir_path}")
        return str(path)

    def apply(self) -> None:
        """Apply the logging configuration, honoring env overrides."""
        # Override from environment variables if provided
        env_level = os.getenv("LOG_LEVEL")
        env_format = os.getenv("LOG_FORMAT")
        env_path = os.getenv("LOG_PATH")

        if env_level:
            self.level = to_log_level(env_level)
        if env_format:
            self.format = env_format
        if env_path:
            self.path = env_path

        # Re-validate after env overrides
        LogConf.model_validate(self.model_dump())

        logger.info(
            "applying log configuration: level=%s, format=%s, path=%s",
            self.level.value,
            self.format,
            self.path,
        )

        handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
        if self.path:
            file_handler = logging.FileHandler(self.path)
            file_handler.setFormatter(logging.Formatter(self.format))
            handlers.append(file_handler)

        logging.basicConfig(
            level=getattr(logging, self.level.value),
            format=self.format,
            handlers=handlers,
            force=True,
        )


class RerankerType(Enum):
    """Enumeration of supported reranker implementations."""

    IDENTITY = "identity"
    BM25 = "bm25"
    RRF_HYBRID = "rrf-hybrid"


class Configuration:
    """Placeholder configuration loader for legacy compatibility."""

    def __init__(self) -> None:
        """Initialize the configuration container."""

    def load(self, config_file: str | None = None) -> None:
        """Load configuration from environment or provided path."""
        load_dotenv()
        env_conf = os.environ["MEMORY_CONFIG"]
        if config_file is None and env_conf:
            config_file = env_conf

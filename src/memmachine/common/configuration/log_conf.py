import logging
import os
import sys
from enum import Enum

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def to_log_level(level_str: str) -> LogLevel:
    try:
        return LogLevel[level_str.upper()]
    except KeyError:
        raise ValueError(f"Invalid log level: {level_str}") from None


class LogConf(BaseModel):
    level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level",
    )
    format: str = Field(
        default="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        description="Logging format string",
    )
    path: str = Field(
        default="/tmp/MemMachine.log",
        description="Path to log file (if empty, logs only to stdout).",
    )

    @field_validator("level", mode="before")
    @staticmethod
    def validate_level(v):
        if isinstance(v, LogLevel):
            return v
        try:
            return to_log_level(str(v))
        except ValueError as e:
            raise ValueError(f"Invalid log level: {v}") from e

    @field_validator("format")
    @staticmethod
    def validate_format(v):
        # A minimal sanity check: must include %(levelname)s and %(message)s
        required_tokens = ["%(levelname)s", "%(message)s", "%(asctime)s"]
        for token in required_tokens:
            if token not in v:
                raise ValueError(f"log format must include {token}, got '{v}'")
        return v

    @field_validator("path")
    @staticmethod
    def validate_path(v):
        if v is None or v == "":
            return None
        # Ensure directory exists and writable
        dir_name = os.path.dirname(v) or "."
        if not os.path.exists(dir_name):
            raise ValueError(f"Log directory does not exist: {dir_name}")
        if not os.access(dir_name, os.W_OK):
            raise ValueError(f"Log directory is not writable: {dir_name}")
        return v

    def apply(self):
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

        handlers = [logging.StreamHandler(sys.stdout)]
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


class ProfilePromptConf(BaseModel):
    profile: str = "profile_prompt"


class RerankerType(Enum):
    IDENTITY = "identity"
    BM25 = "bm25"
    RRF_HYBRID = "rrf-hybrid"


class Configuration:
    def __init__(self):
        pass

    def load(self, config_file: str | None = None):
        load_dotenv()
        env_conf = os.environ["MEMORY_CONFIG"]
        if config_file is None and env_conf:
            config_file = env_conf

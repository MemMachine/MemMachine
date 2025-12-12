"""Configuration models and helpers for MemMachine runtime."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, TypeGuard, cast

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from memmachine.common.configuration.database_conf import DatabasesConf
from memmachine.common.configuration.embedder_conf import EmbeddersConf
from memmachine.common.configuration.episodic_config import (
    EpisodicMemoryConfPartial,
)
from memmachine.common.configuration.language_model_conf import LanguageModelsConf
from memmachine.common.configuration.log_conf import LogConf
from memmachine.common.configuration.mixin_confs import YamlSerializableMixin
from memmachine.common.configuration.reranker_conf import RerankersConf
from memmachine.common.errors import (
    DefaultEmbedderNotConfiguredError,
    DefaultRerankerNotConfiguredError,
    EmbedderNotFoundError,
    RerankerNotFoundError,
)
from memmachine.semantic_memory.semantic_model import SemanticCategory
from memmachine.semantic_memory.semantic_session_manager import IsolationType
from memmachine.server.prompt.default_prompts import PREDEFINED_SEMANTIC_CATEGORIES

YamlValue = dict[str, "YamlValue"] | list["YamlValue"] | str | int | float | bool | None


logger = logging.getLogger(__name__)


class SessionManagerConf(YamlSerializableMixin):
    """Configuration for the session database connection."""

    database: str = Field(
        default="",
        description="The database ID to use for session manager",
    )


class EpisodeStoreConf(YamlSerializableMixin):
    """Configuration for the episode storage."""

    database: str = Field(
        default="",
        description="The database ID to use for episode storage",
    )
    with_count_cache: bool = Field(
        default=True,
        description="Whether to use a in memory cache for counting messages per session.",
    )


class SemanticMemoryConf(YamlSerializableMixin):
    """Configuration for semantic memory defaults."""

    database: str = Field(
        ...,
        description="The database to use for semantic memory",
    )
    llm_model: str = Field(
        ...,
        description="The default language model to use for semantic memory",
    )
    embedding_model: str = Field(
        ...,
        description="The embedding model to use for semantic memory",
    )


def _read_txt(filename: str) -> str:
    """Read a text file into a string, resolving relative paths from CWD."""
    path = Path(filename)
    if not path.is_absolute():
        path = Path.cwd() / path

    with path.open("r", encoding="utf-8") as f:
        return f.read()


class PromptConf(YamlSerializableMixin):
    """Prompt configuration for semantic memory contexts."""

    profile: list[str] = Field(
        default=["profile_prompt", "writing_assistant_prompt", "coding_prompt"],
        description="The default prompts to use for semantic user memory",
    )
    role: list[str] = Field(
        default=[],
        description="The default prompts to use for semantic role memory",
    )
    session: list[str] = Field(
        default=["profile_prompt"],
        description="The default prompts to use for semantic session memory",
    )
    episode_summary_system_prompt_path: str = Field(
        default="",
        description="The prompt template to use for episode summary generation - system part",
    )
    episode_summary_user_prompt_path: str = Field(
        default="",
        description="The prompt template to use for episode summary generation - user part",
    )

    @classmethod
    def prompt_exists(cls, prompt_name: str) -> bool:
        """Return True if the prompt name is known."""
        return prompt_name in PREDEFINED_SEMANTIC_CATEGORIES

    @field_validator("profile", "session", "role", check_fields=True)
    @classmethod
    def validate_profile(cls, v: list[str]) -> list[str]:
        """Validate that provided prompts exist."""
        for prompt_name in v:
            if not cls.prompt_exists(prompt_name):
                raise ValueError(f"Prompt {prompt_name} does not exist")
        return v

    @property
    def episode_summary_system_prompt(self) -> str:
        """Load the system portion of the episode summary prompt."""
        file_path = self.episode_summary_system_prompt_path
        if not file_path:
            txt = "default_episode_summary_system_prompt.txt"
            file_path = str(Path(__file__).parent / txt)
        return _read_txt(file_path)

    @property
    def episode_summary_user_prompt(self) -> str:
        """Load the user portion of the episode summary prompt."""
        file_path = self.episode_summary_user_prompt_path
        if not file_path:
            txt = "default_episode_summary_user_prompt.txt"
            file_path = str(Path(__file__).parent / txt)
        return _read_txt(file_path)

    @property
    def default_semantic_categories(
        self,
    ) -> dict[IsolationType, list[SemanticCategory]]:
        """Build the default semantic categories for each isolation type."""
        semantic_categories = PREDEFINED_SEMANTIC_CATEGORIES

        return {
            IsolationType.SESSION: [
                semantic_categories[s_name] for s_name in self.session
            ],
            IsolationType.ROLE: [semantic_categories[s_name] for s_name in self.role],
            IsolationType.USER: [
                semantic_categories[s_name] for s_name in self.profile
            ],
        }


class ResourcesConf(BaseModel):
    """Configuration for MemMachine common resources."""

    embedders: EmbeddersConf
    language_models: LanguageModelsConf
    rerankers: RerankersConf
    databases: DatabasesConf

    def to_yaml_dict(self) -> dict:
        return {
            "embedders": self.embedders.to_yaml_dict(),
            "language_models": self.language_models.to_yaml_dict(),
            "rerankers": self.rerankers.to_yaml_dict(),
            "databases": self.databases.to_yaml_dict(),
        }

    def to_yaml(self) -> str:
        """Serialize the resources configuration to a YAML string."""
        data = self.to_yaml_dict()
        return yaml.safe_dump(data, sort_keys=True)

    @model_validator(mode="before")
    @classmethod
    def parse(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data

        embedders = EmbeddersConf.parse(data)
        language_models = LanguageModelsConf.parse(data)
        rerankers = RerankersConf.parse(data)
        databases = DatabasesConf.parse(data)

        return {
            "embedders": embedders,
            "language_models": language_models,
            "rerankers": rerankers,
            "databases": databases,
        }


class ServerConf(YamlSerializableMixin):
    """Configuration for MemMachine API server settings."""

    host: str = Field(
        default="localhost",
        description="The host address for the MemMachine API server",
    )
    port: int = Field(
        default=8080,
        description="The port number for the MemMachine API server",
        gt=0,
        lt=65536,
    )

    @model_validator(mode="before")
    @classmethod
    def _overwrite_with_env_variable(cls, data: dict) -> dict:
        data = dict(data or {})

        host = os.getenv("HOST")
        if host:
            data["host"] = host

        port = os.getenv("PORT")
        if port:
            data["port"] = port

        return data


class Configuration(BaseModel):
    """Aggregate configuration for MemMachine services."""

    episodic_memory: EpisodicMemoryConfPartial
    semantic_memory: SemanticMemoryConf
    logging: LogConf
    prompt: PromptConf = PromptConf()
    session_manager: SessionManagerConf
    resources: ResourcesConf
    episode_store: EpisodeStoreConf
    server: ServerConf = ServerConf()

    def check_reranker(self, reranker_name: str) -> None:
        long_term_memory = self.episodic_memory.long_term_memory
        if not reranker_name or not long_term_memory:
            raise DefaultRerankerNotConfiguredError
        if not self.resources.rerankers.contains_reranker(reranker_name):
            raise RerankerNotFoundError(reranker_name)

    @property
    def default_long_term_memory_embedder(self) -> str:
        long_term_memory = self.episodic_memory.long_term_memory
        if not long_term_memory or not long_term_memory.embedder:
            raise DefaultEmbedderNotConfiguredError
        return long_term_memory.embedder

    def check_embedder(self, embedder_name: str) -> None:
        long_term_memory = self.episodic_memory.long_term_memory
        if not embedder_name or not long_term_memory:
            raise DefaultEmbedderNotConfiguredError
        if not self.resources.embedders.contains_embedder(embedder_name):
            raise EmbedderNotFoundError(embedder_name)

    @property
    def default_long_term_memory_reranker(self) -> str:
        long_term_memory = self.episodic_memory.long_term_memory
        if not long_term_memory or not long_term_memory.reranker:
            raise DefaultRerankerNotConfiguredError
        return long_term_memory.reranker

    def to_yaml(self) -> str:
        data = {
            "episodic_memory": self.episodic_memory.to_yaml_dict(),
            "semantic_memory": self.semantic_memory.to_yaml_dict(),
            "logging": self.logging.to_yaml_dict(),
            "prompt": self.prompt.to_yaml_dict(),
            "session_manager": self.session_manager.to_yaml_dict(),
            "resources": self.resources.to_yaml_dict(),
            "episode_store": self.episode_store.to_yaml_dict(),
            "server": self.server.to_yaml_dict(),
        }
        return yaml.safe_dump(data, sort_keys=True)

    @classmethod
    def load_yml_file(cls, config_file: str) -> Configuration:
        """Load configuration from a YAML file path."""
        config_path = Path(config_file)
        logger.info("Loading configuration from '%s'", config_file)
        try:
            yaml_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as err:
            raise FileNotFoundError(f"Config file {config_file} not found") from err
        except yaml.YAMLError as err:
            raise ValueError(f"Config file {config_file} is not valid YAML") from err
        except Exception as err:
            raise RuntimeError(f"Failed to load config file {config_file}") from err

        def config_to_lowercase(data: YamlValue) -> YamlValue:
            """Recursively convert dictionary keys in a nested structure to lowercase."""
            if isinstance(data, dict):
                return {k.lower(): config_to_lowercase(v) for k, v in data.items()}
            if isinstance(data, list):
                return [config_to_lowercase(i) for i in data]
            return data

        yaml_config = config_to_lowercase(yaml_config)

        def is_mapping(val: YamlValue) -> TypeGuard[dict[str, YamlValue]]:
            return isinstance(val, dict)

        if not is_mapping(yaml_config):
            raise TypeError(f"Root of YAML config '{config_path}' must be a mapping")

        mapping_config = cast(dict[str, Any], yaml_config)

        return Configuration(**mapping_config)

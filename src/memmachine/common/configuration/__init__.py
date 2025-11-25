"""Configuration models and helpers for MemMachine runtime."""

from __future__ import annotations

from pathlib import Path
from typing import TypeGuard, cast

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from memmachine.common.configuration.database_conf import DatabasesConf
from memmachine.common.configuration.embedder_conf import EmbeddersConf
from memmachine.common.configuration.episodic_config import (
    EpisodicMemoryConfPartial,
)
from memmachine.common.configuration.language_model_conf import LanguageModelsConf
from memmachine.common.configuration.log_conf import LogConf
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


class SessionManagerConf(BaseModel):
    """Configuration for the session database connection."""

    database: str = Field(
        default="",
        description="The database ID to use for session manager",
    )


class EpisodeStoreConf(BaseModel):
    """Configuration for the episod storage."""

    database: str = Field(
        default="",
        description="The database ID to use for episode storage",
    )


class SemanticMemoryConf(BaseModel):
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


class PromptConf(BaseModel):
    """Prompt configuration for semantic memory contexts."""

    profile: list[str] = Field(
        default=["profile_prompt", "writing_assistant_prompt"],
        description="The default prompts to use for semantic user memory",
    )
    role: list[str] = Field(
        default=[],
        description="The default prompts to use for semantic role memory",
    )
    session: list[str] = Field(
        default=[],
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


class Configuration(BaseModel):
    """Aggregate configuration for MemMachine services."""

    episodic_memory: EpisodicMemoryConfPartial
    semantic_memory: SemanticMemoryConf
    logging: LogConf
    prompt: PromptConf = PromptConf()
    session_manager: SessionManagerConf
    resources: ResourcesConf
    episode_store: EpisodeStoreConf

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


def load_config_yml_file(config_file: str) -> Configuration:
    """Load configuration from a YAML file path."""
    config_path = Path(config_file)
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

    mapping_config = cast(dict[str, YamlValue], yaml_config)

    return Configuration(**mapping_config)

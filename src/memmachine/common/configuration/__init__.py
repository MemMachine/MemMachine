import os.path
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Self

import yaml
from pydantic import BaseModel, Field, root_validator, model_validator

from ...common.configuration.embedder_conf import EmbedderConf
from ...common.configuration.log_conf import LogConf
from ...common.configuration.model_conf import LanguageModelConf
from ...common.configuration.reranker_conf import RerankerConf
from ...common.configuration.storage_conf import StorageConf


class SessionDBConf(BaseModel):
    uri: str = Field(
        default="sqlitetest.db",
        description="Database URI",
    )
    storage_id: str = Field(
        default="",
        description="The storage ID to use for session DB",
    )


class LongTermMemoryConfPartial(BaseModel):
    """The partial configuration for LongTermMemoryConf.

    All fields are optional. Used for updates."""

    embedder: str | None = Field(
        default=None,
        description="The embedder to use for long-term memory",
    )
    reranker: str | None = Field(
        default=None,
        description="The reranker to use for long-term memory",
    )
    vector_graph_store: str | None = Field(
        default=None,
        description="The vector graph store to use for long-term memory",
    )
    enabled: bool | None = Field(
        default=None,
        description="Whether long-term memory is enabled",
    )


class LongTermMemoryConf(BaseModel):
    embedder: str = Field(
        ...,
        description="The embedder to use for long-term memory",
    )
    reranker: str = Field(
        ...,
        description="The reranker to use for long-term memory",
    )
    vector_graph_store: str = Field(
        ...,
        description="The vector graph store to use for long-term memory",
    )
    enabled: bool = Field(
        default=True,
        description="Whether long-term memory is enabled",
    )

    def update(self, other: LongTermMemoryConfPartial) -> Self:
        """Return a new configuration with fields updated from a partial config."""
        update_data = {
            k: v
            for k, v in other.model_dump(exclude_unset=True).items()
            if v is not None
        }
        # Use Pydantic's built-in helper for safe, validated merging
        return self.model_copy(update=update_data)


class ProfileMemoryConf(BaseModel):
    llm_model: str = Field(
        ...,
        description="The language model to use for profile memory",
    )
    embedding_model: str = Field(
        ...,
        description="The embedding model to use for profile memory",
    )
    database: str = Field(
        ...,
        description="The database to use for profile memory",
    )
    prompt: str = Field(
        ...,
        description="The prompt template to use for profile memory",
    )


class SessionMemoryConfPartial(BaseModel):
    model_name: str | None = Field(
        default=None,
        description="The language model to use for session memory",
    )
    message_capacity: int | None = Field(
        default=None,
        description="The maximum number of messages to retain in session memory",
        gt=0,
    )
    max_message_length: int | None = Field(
        default=None,
        description="The maximum length of each message in characters",
        gt=0,
    )
    max_token_num: int | None = Field(
        default=None,
        description="The maximum number of tokens to retain in session memory",
        gt=0,
    )
    enabled: bool | None = Field(
        default=None,
        description="Whether session memory is enabled",
    )


class SessionMemoryConf(BaseModel):
    model_name: str = Field(
        ...,
        description="The language model to use for session memory",
    )
    message_capacity: int = Field(
        default=500,
        description="The maximum number of messages to retain in session memory",
        gt=0,
    )
    max_message_length: int = Field(
        default=16000,
        description="The maximum length of each message in characters",
        gt=0,
    )
    max_token_num: int = Field(
        default=8000,
        description="The maximum number of tokens to retain in session memory",
        gt=0,
    )
    enabled: bool = Field(
        default=True,
        description="Whether session memory is enabled",
    )

    def update(self, other: SessionMemoryConfPartial) -> Self:
        """Return a new configuration with fields updated from a partial config."""
        update_data = {
            k: v
            for k, v in other.model_dump(exclude_unset=True).items()
            if v is not None
        }
        # Use Pydantic's built-in helper for safe, validated merging
        return self.model_copy(update=update_data)


def _read_txt(filename: str) -> str:
    """
    Reads a text file and returns its contents as a string.

    Behavior:
      - If `filename` is an absolute path, read it directly.
      - If `filename` is a relative path, read it from the current working directory.

    Args:
        filename (str): File name or absolute path. Optional.

    Returns:
        str: The file's content as a string.
    """
    path = Path(filename)
    if not path.is_absolute():
        path = Path.cwd() / path

    with path.open("r", encoding="utf-8") as f:
        return f.read()


class ProfilePrompt(BaseModel):
    update_prompt: str = Field(
        ...,
        description="The prompt template to use for profile update",
    )
    consolidation_prompt: str = Field(
        ...,
        description="The prompt template to use for profile consolidation",
    )

    @staticmethod
    def load_from_module(prompt_module: ModuleType):
        update_prompt = getattr(prompt_module, "UPDATE_PROMPT", "")
        consolidation_prompt = getattr(prompt_module, "CONSOLIDATION_PROMPT", "")

        return ProfilePrompt(
            update_prompt=update_prompt,
            consolidation_prompt=consolidation_prompt,
        )


class PromptConf(BaseModel):
    profile: str = Field(
        default="profile_prompt",
        description="The prompt template to use for profile memory",
    )
    episode_summary_system_prompt_path: str = Field(
        default="",
        description="The prompt template to use for episode summary generation - system part",
    )
    episode_summary_user_prompt_path: str = Field(
        default="",
        description="The prompt template to use for episode summary generation - user part",
    )

    @property
    def episode_summary_system_prompt(self) -> str:
        file_path = self.episode_summary_system_prompt_path
        if not file_path:
            txt = "dft_episode_summary_system_prompt.txt"
            file_path = os.path.join(Path(__file__).parent, txt)
        return _read_txt(file_path)

    @property
    def episode_summary_user_prompt(self) -> str:
        file_path = self.episode_summary_user_prompt_path
        if not file_path:
            txt = "dft_episode_summary_user_prompt.txt"
            file_path = os.path.join(Path(__file__).parent, txt)
        return _read_txt(file_path)

    @property
    def profile_prompt(self) -> "ProfilePrompt":
        prompt_package = "memmachine.server.prompt"
        module_name = f"{prompt_package}.{self.profile}"

        try:
            prompt_module: ModuleType = import_module(module_name)
        except ModuleNotFoundError as e:
            raise ImportError(
                f"Prompt profile '{self.profile}' not found in package '{prompt_package}'."
            ) from e

        try:
            prompt = ProfilePrompt.load_from_module(prompt_module)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load prompt from module '{module_name}': {e}"
            ) from e
        return prompt


class EpisodicMemoryConfPartial(BaseModel):
    sessionMemory: SessionMemoryConfPartial | None = Field(
        default=None,
        description="Partial configuration for session memory in episodic memory",
    )
    long_term_memory: LongTermMemoryConfPartial | None = Field(
        default=None,
        description="Partial configuration for long-term memory in episodic memory",
    )


class EpisodicMemoryConf(BaseModel):
    sessionmemory: SessionMemoryConf = Field(
        ...,
        description="Configuration for session memory in episodic memory",
    )
    long_term_memory: LongTermMemoryConf = Field(
        ...,
        description="Configuration for long-term memory in episodic memory",
    )

    def update(self, other: EpisodicMemoryConfPartial) -> Self:
        """Return a new configuration with fields updated from a partial config."""
        update_data: dict[str, Any] = {}
        if other.sessionMemory is not None:
            update_data["sessionMemory"] = self.sessionMemory.update(
                other.sessionMemory
            )
        if other.long_term_memory is not None:
            update_data["long_term_memory"] = self.long_term_memory.update(
                other.long_term_memory
            )
        # Use Pydantic's built-in helper for safe, validated merging
        return self.model_copy(update=update_data)


class Configuration(EpisodicMemoryConf):
    logging: LogConf
    sessiondb: SessionDBConf
    model: LanguageModelConf
    storage: StorageConf
    profile_memory: ProfileMemoryConf
    embedder: EmbedderConf
    reranker: RerankerConf
    prompt: PromptConf

    def __init__(self, **data):
        data = data.copy()  # avoid mutating caller's dict
        super().__init__(**data)
        self.model = LanguageModelConf.parse_language_model_conf(data)
        self.storage = StorageConf.parse_storage_conf(data)
        self.embedder = EmbedderConf.parse_embedder_conf(data)
        self.reranker = RerankerConf.parse_reranker_conf(data)


def load_config_yml_file(config_file: str) -> Configuration:
    try:
        yaml_config = yaml.safe_load(open(config_file, encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {config_file} not found")
    except yaml.YAMLError:
        raise ValueError(f"Config file {config_file} is not valid YAML")
    except Exception as e:
        raise e

    def config_to_lowercase(data: Any) -> Any:
        """Recursively converts all dictionary keys in a nested structure
        to lowercase."""
        if isinstance(data, dict):
            return {k.lower(): config_to_lowercase(v) for k, v in data.items()}
        if isinstance(data, list):
            return [config_to_lowercase(i) for i in data]
        return data

    yaml_config = config_to_lowercase(yaml_config)
    return Configuration(**yaml_config)

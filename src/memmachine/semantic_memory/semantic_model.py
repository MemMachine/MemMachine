"""Core data models for semantic memory features and retrieval."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, InstanceOf

from memmachine.common.embedder import Embedder
from memmachine.common.episode_store import EpisodeIdT
from memmachine.common.language_model import LanguageModel
from memmachine.semantic_memory.util import semantic_prompt_template

SetIdT = str
FeatureIdT = str


class SemanticCommandType(Enum):
    """Semantic memory actions that can be applied to a feature."""

    ADD = "add"
    DELETE = "delete"


class SemanticCommand(BaseModel):
    """Normalized instruction emitted by the LLM to mutate semantic features."""

    command: SemanticCommandType
    feature: str
    tag: str
    value: str


@dataclass
class RawSemanticPrompt:
    """Pair of prompt templates driving update and consolidation LLM calls."""

    update_prompt: str
    consolidation_prompt: str


class StructuredSemanticPrompt(BaseModel):
    """Pair of prompt templates driving update and consolidation LLM calls."""

    tags: dict[str, str]
    description: str | None = None

    @property
    def update_prompt(self) -> str:
        return semantic_prompt_template.build_update_prompt(
            tags=self.tags,
            description=self.description if self.description else "",
        )

    @property
    def consolidation_prompt(self) -> str:
        return semantic_prompt_template.build_consolidation_prompt()


class SemanticFeature(BaseModel):
    """Semantic memory entry composed of category, tag, feature name, and textual value."""

    class Metadata(BaseModel):
        """Storage metadata for a semantic feature, including id and citations."""

        citations: list[EpisodeIdT] | None = None
        id: FeatureIdT | None = None
        other: dict[str, Any] | None = None

    set_id: SetIdT | None = None
    category: str
    tag: str
    feature_name: str
    value: str
    metadata: Metadata = Metadata()

    @staticmethod
    def group_features(
        features: list["SemanticFeature"],
    ) -> dict[tuple[str, str, str], list["SemanticFeature"]]:
        grouped_features: dict[tuple[str, str, str], list[SemanticFeature]] = {}

        for f in features:
            key = (f.category, f.tag, f.feature_name)

            if key not in grouped_features:
                grouped_features[key] = []

            grouped_features[key].append(f)

        return grouped_features

    @staticmethod
    def group_features_by_tag(
        features: list["SemanticFeature"],
    ) -> dict[str, list["SemanticFeature"]]:
        grouped_features: dict[str, list[SemanticFeature]] = {}

        for f in features:
            key = f.tag

            if key not in grouped_features:
                grouped_features[key] = []

            grouped_features[key].append(f)

        return grouped_features


@runtime_checkable
class SemanticPrompt(Protocol):
    """Protocol describing prompt templates for semantic extraction."""

    @property
    def update_prompt(self) -> str:
        raise NotImplementedError

    @property
    def consolidation_prompt(self) -> str:
        raise NotImplementedError


class SemanticCategory(BaseModel):
    """Defines a semantic feature category, its allowed tags, and prompt strategy."""

    id: int | None = None

    name: str
    prompt: InstanceOf[SemanticPrompt]


class Resources(BaseModel):
    """Resource bundle (embedder, language model, semantic categories) for a set_id."""

    embedder: InstanceOf[Embedder]
    language_model: InstanceOf[LanguageModel]
    semantic_categories: list[InstanceOf[SemanticCategory]]


@runtime_checkable
class ResourceRetriever(Protocol):
    """Protocol for locating the `Resources` bundle associated with a set_id."""

    def get_resources(self, set_id: SetIdT) -> Resources:
        raise NotImplementedError

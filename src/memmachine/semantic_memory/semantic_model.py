from dataclasses import dataclass
from datetime import datetime
from types import ModuleType
from typing import Any, Optional, Protocol, Tuple, runtime_checkable

from pydantic import BaseModel, InstanceOf

from memmachine.common.embedder import Embedder
from memmachine.common.language_model import LanguageModel


class SemanticCommand(BaseModel):
    command: str
    feature: str
    tag: str
    value: str


class HistoryMessage(BaseModel):
    class Metadata(BaseModel):
        id: Optional[int] = None
        other: Optional[dict[str, Any]] = None

    content: str
    created_at: datetime

    metadata: Metadata = Metadata()


@dataclass
class SemanticPrompt:
    update_prompt: str
    consolidation_prompt: str

    @staticmethod
    def load_from_module(prompt_module: ModuleType):
        update_prompt = getattr(prompt_module, "UPDATE_PROMPT", "")
        consolidation_prompt = getattr(prompt_module, "CONSOLIDATION_PROMPT", "")

        return SemanticPrompt(
            update_prompt=update_prompt,
            consolidation_prompt=consolidation_prompt,
        )


class SemanticFeature(BaseModel):
    class Metadata(BaseModel):
        citations: Optional[list[HistoryMessage]] = None
        id: Optional[int] = None
        other: Optional[dict[str, Any]] = None

    set_id: Optional[str] = None
    type: str
    tag: str
    feature: str
    value: str
    metadata: Metadata = Metadata()

    @staticmethod
    def group_features(
        features: list["SemanticFeature"],
    ) -> dict[Tuple[str, str, str], list["SemanticFeature"]]:
        grouped_features: dict[Tuple[str, str, str], list[SemanticFeature]] = {}

        for f in features:
            key = (f.type, f.tag, f.feature)

            if key not in grouped_features:
                grouped_features[key] = []

            grouped_features[key].append(f)

        return grouped_features

    @staticmethod
    def group_features_by_tag(
        features: list["SemanticFeature"],
    ) -> dict[Tuple[str, str], list["SemanticFeature"]]:
        grouped_features: dict[Tuple[str, str], list[SemanticFeature]] = {}

        for f in features:
            key = (f.tag, f.feature)

            if key not in grouped_features:
                grouped_features[key] = []

            grouped_features[key].append(f)

        return grouped_features


class SemanticType(BaseModel):
    id: Optional[int] = None

    name: str
    tags: set[str]
    prompt: SemanticPrompt


@runtime_checkable
class ResourceRetriever(Protocol):
    class Resources(BaseModel):
        embedder: InstanceOf[Embedder]
        language_model: InstanceOf[LanguageModel]
        semantic_types: list[InstanceOf[SemanticType]]

    def get_resources(self, set_id: str) -> Resources:
        raise NotImplementedError

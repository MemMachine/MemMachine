"""LLM helpers for extracting and consolidating semantic features."""

import json
import logging

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    TypeAdapter,
    validate_call,
)

from memmachine.common.episode_store import EpisodeIdT
from memmachine.common.language_model import LanguageModel
from memmachine.semantic_memory.semantic_model import SemanticCommand, SemanticFeature

logger = logging.getLogger(__name__)


def _features_to_llm_format(
    features: list[SemanticFeature],
) -> dict[str, dict[str, str]]:
    structured_features: dict[str, dict[str, str]] = {}

    for feature in features:
        if feature.tag not in structured_features:
            structured_features[feature.tag] = {}

        structured_features.setdefault(feature.tag, {})[feature.feature_name] = (
            feature.value
        )

    return structured_features


class _SemanticFeatureUpdateRes(BaseModel):
    """Schema used to validate parsed feature-update commands returned by the LLM."""

    commands: list[SemanticCommand] = Field(default_factory=list)


@validate_call
async def llm_feature_update(
    features: list[SemanticFeature],
    message_content: str,
    model: InstanceOf[LanguageModel],
    update_prompt: str,
) -> list[SemanticCommand]:
    """Generate feature update commands from an incoming message using the LLM."""
    user_prompt = (
        "The old feature set is provided below:\n"
        "<OLD_PROFILE>\n"
        f"{json.dumps(_features_to_llm_format(features))}\n"
        "</OLD_PROFILE>\n"
        "\n"
        "The history is provided below:\n"
        "<HISTORY>\n"
        f"{message_content}\n"
        "</HISTORY>\n"
    )

    parsed_output = await model.generate_parsed_response(
        system_prompt=update_prompt,
        user_prompt=user_prompt,
        output_format=_SemanticFeatureUpdateRes,
    )

    if parsed_output is None:
        return []

    validated_output = TypeAdapter(_SemanticFeatureUpdateRes).validate_python(
        parsed_output,
    )
    return validated_output.commands


class LLMReducedFeature(BaseModel):
    """Minimal feature payload emitted by the consolidation prompt for reinsertion."""

    tag: str
    feature: str
    value: str


class SemanticConsolidateMemoryRes(BaseModel):
    """LLM response describing merged features and ids of features to retain."""

    consolidated_memories: list[LLMReducedFeature] = Field(default_factory=list)
    keep_memories: list[EpisodeIdT] | None
    model_config = ConfigDict(coerce_numbers_to_str=True)


@validate_call
async def llm_consolidate_features(
    features: list[SemanticFeature],
    model: InstanceOf[LanguageModel],
    consolidate_prompt: str,
) -> SemanticConsolidateMemoryRes | None:
    """Merge overlapping features and return consolidation commands from the LLM."""
    parsed_output = await model.generate_parsed_response(
        system_prompt=consolidate_prompt,
        user_prompt=json.dumps(_features_to_llm_format(features)),
        output_format=SemanticConsolidateMemoryRes,
    )

    if parsed_output is None:
        return None

    validated_output = TypeAdapter(SemanticConsolidateMemoryRes).validate_python(
        parsed_output,
    )
    return validated_output

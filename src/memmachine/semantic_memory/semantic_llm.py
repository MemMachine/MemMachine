import json
import logging
from typing import Any

from pydantic import (
    BaseModel,
    InstanceOf,
    ValidationError,
    field_validator,
    validate_call,
)

from memmachine.common.data_types import ExternalServiceAPIError
from memmachine.common.language_model import LanguageModel
from memmachine.semantic_memory.semantic_model import SemanticCommand, SemanticFeature

logger = logging.getLogger(__name__)


def _process_commands(commands) -> list[SemanticCommand]:
    valid_commands = []
    for command in commands:
        if not isinstance(command, dict):
            logger.warning(
                "AI response format incorrect: "
                "expected feature update command to be dict, got %s %s",
                type(command).__name__,
                command,
            )
            continue

        try:
            pydantic_command = SemanticCommand(**command)
        except Exception as e:
            logger.warning(
                "AI response format incorrect: unable to parse feature update command %s, error %s",
                command,
                str(e),
            )
            continue

        valid_commands.append(pydantic_command)

    return valid_commands


async def _llm_think(
    model: LanguageModel,
    system_prompt: str,
    user_prompt: str,
):
    try:
        response_text, _ = await model.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except (ExternalServiceAPIError, ValueError, RuntimeError) as e:
        raise Exception("Model Error when processing semantic features") from e

    # Get thinking and JSON from language model response.
    thinking, _, response_json = response_text.removeprefix("<think>").rpartition(
        "</think>"
    )
    thinking = thinking.strip()

    try:
        response = json.loads(response_json)
    except ValueError as e:
        raise ValueError(
            "Unable to load language model output '%s' as JSON, Error %s",
            str(response_json),
            e,
        )

    return thinking, response


def _features_to_llm_format(
    features: list[SemanticFeature],
) -> dict[str, dict[str, str]]:
    structured_features = {}

    for feature in features:
        if feature.tag not in structured_features:
            structured_features[feature.tag] = {}

        structured_features[feature.tag][feature.feature] = feature.value

    return structured_features


@validate_call
async def llm_feature_update(
    features: list[SemanticFeature],
    message_content: str,
    model: InstanceOf[LanguageModel],
    update_prompt: str,
) -> list[SemanticCommand]:
    user_prompt = (
        "The old feature set is provided below:\n"
        "<OLD_PROFILE>\n"
        "{feature_set}\n"
        "</OLD_PROFILE>\n"
        "\n"
        "The history is provided below:\n"
        "<HISTORY>\n"
        "{message_content}\n"
        "</HISTORY>\n"
    ).format(
        feature_set=json.dumps(_features_to_llm_format(features)),
        message_content=message_content,
    )

    thinking, raw_commands = await _llm_think(
        model=model,
        system_prompt=update_prompt,
        user_prompt=user_prompt,
    )

    logger.debug(
        "PROFILE MEMORY INGESTOR",
        extra={
            "queries_to_ingest": message_content,
            "thoughts": thinking,
            "outputs": raw_commands,
        },
    )

    # This should probably just be a list of commands
    # instead of a dictionary mapping
    # from integers in strings (not even bare ints!)
    # to commands.
    # TODO: Consider improving this design in a breaking change.
    if not isinstance(raw_commands, dict):
        logger.warning(
            "AI response format incorrect: expected dict, got %s %s",
            type(raw_commands).__name__,
            raw_commands,
        )
        return []

    return _process_commands(raw_commands.values())


class SemanticConsolidateMemoryRes(BaseModel):
    consolidated_memories: list[SemanticFeature]
    keep_memories: list[int] | None

    @field_validator("consolidated_memories", mode="before")
    @classmethod
    def _filter_and_validate_memories(cls, v: Any) -> list[SemanticFeature]:
        if v is None:
            return []
        if not isinstance(v, list):
            logger.warning(
                "AI response format incorrect: 'consolidate_memories' not a list, got %s %s",
                type(v).__name__,
                v,
            )
            return []

        cleaned: list[SemanticFeature] = []
        for i, item in enumerate(v):
            try:
                cleaned.append(SemanticFeature.model_validate(item))
            except ValidationError as e:
                logger.warning(
                    "Dropping invalid memory at index %d. Error: %s. Item: %r",
                    i,
                    e.errors(),
                    item,
                )
        return cleaned

    @field_validator("keep_memories", mode="before")
    @classmethod
    def _normalize_keep_memories(cls, v: Any) -> list[int] | None:
        if v is None:
            return None
        if not isinstance(v, list):
            logger.warning(
                "AI response format incorrect: 'keep_memories' not a list, got %s %s",
                type(v).__name__,
                v,
            )
            return []
        ints: list[int] = []
        for i, x in enumerate(v):
            try:
                ints.append(int(x))
            except Exception:
                logger.warning(
                    "Dropping invalid keep_memories[%d]=%r (not coercible to int)", i, x
                )
        return ints


# @validate_call
async def llm_consolidate_features(
    features: list[SemanticFeature],
    model: InstanceOf[LanguageModel],
    consolidate_prompt: str,
) -> SemanticConsolidateMemoryRes | None:
    try:
        thinking, updated_feature_entries = await _llm_think(
            model=model,
            system_prompt=consolidate_prompt,
            user_prompt=json.dumps(_features_to_llm_format(features)),
        )
    except ValueError as e:
        logger.exception("Unable to consolidate features from LLM")
        raise e

    logger.debug(
        "PROFILE MEMORY CONSOLIDATOR",
        extra={
            "receives": features,
            "thoughts": thinking,
            "outputs": updated_feature_entries,
        },
    )

    if not isinstance(updated_feature_entries, dict):
        logger.warning(
            "AI response format incorrect: expected dict, got %s %s",
            type(updated_feature_entries).__name__,
            updated_feature_entries,
        )
        return None

    try:
        return SemanticConsolidateMemoryRes.model_validate(updated_feature_entries)
    except ValidationError:
        logger.warning(
            "AI response format incorrect: expected dict with keys 'consolidated_memories' and 'keep_memories', got %s %s",
            type(updated_feature_entries).__name__,
            updated_feature_entries,
        )
        return None

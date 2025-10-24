import json
import logging
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator, validate_call, InstanceOf

from memmachine.common.data_types import ExternalServiceAPIError
from memmachine.common.language_model import LanguageModel

logger = logging.getLogger(__name__)


class SemanticFeature(BaseModel):
    class Metadata(BaseModel):
        citations: list[int] | None = None
        id: int | None = None

    tag: str
    feature: str
    value: str
    metadata: Metadata = Metadata()


class SemanticCommand(BaseModel):
    command: str
    feature: str
    tag: str
    value: str


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
        logger.error("Model Error when processing semantic features: %s", str(e))
        return

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


@validate_call
async def llm_feature_update(
    features,
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
        feature_set=str(features),
        message_content=message_content,
    )

    try:
        thinking, raw_commands = await _llm_think(
            model=model,
            system_prompt=update_prompt,
            user_prompt=user_prompt,
        )
    except ValueError:
        return []

    logger.info(
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
    consolidate_memories: list[SemanticFeature]
    keep_memories: list[int] | None

    @field_validator("consolidate_memories", mode="before")
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


@validate_call
async def llm_consolidate_features(
    memories: list[SemanticFeature],
    model: InstanceOf[LanguageModel],
    consolidate_prompt: str,
) -> SemanticConsolidateMemoryRes | None:
    try:
        thinking, updated_feature_entries = await _llm_think(
            model=model,
            system_prompt=consolidate_prompt,
            user_prompt=json.dumps(memories),
        )
    except ValueError as e:
        logger.exception("Unable to consolidate features from LLM")
        raise e

    logger.info(
        "PROFILE MEMORY CONSOLIDATOR",
        extra={
            "receives": memories,
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

    return SemanticConsolidateMemoryRes.model_validate(updated_feature_entries)

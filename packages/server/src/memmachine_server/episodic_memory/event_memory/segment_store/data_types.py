"""Data types for segment store."""

import json
from collections.abc import Mapping
from uuid import UUID

from pydantic import BaseModel, Field, JsonValue, TypeAdapter, field_validator

from memmachine_server.common.payload_codec.payload_codec_config import (
    PayloadCodecConfig,
    PlaintextPayloadCodecConfig,
    decode_payload_codec_config,
    encode_payload_codec_config,
)

_JSON_OBJECT_ADAPTER = TypeAdapter(dict[str, JsonValue])


class SegmentStorePartitionConfig(BaseModel):
    """Configuration for a logical partition in a segment store."""

    payload_codec_config: PayloadCodecConfig = Field(
        default_factory=PlaintextPayloadCodecConfig,
        description=("Payload codec configuration for the partition."),
    )

    @field_validator("payload_codec_config", mode="before")
    @classmethod
    def _default_payload_codec_config(cls, value: object) -> object:
        if value is None:
            return PlaintextPayloadCodecConfig()
        if isinstance(value, Mapping):
            return decode_payload_codec_config(
                _JSON_OBJECT_ADAPTER.validate_python(value)
            )
        return value

    @field_validator("payload_codec_config", mode="after")
    @classmethod
    def _validate_payload_codec_config(
        cls, value: PayloadCodecConfig
    ) -> PayloadCodecConfig:
        try:
            json.dumps(encode_payload_codec_config(value))
        except TypeError as err:
            raise ValueError("payload_codec_config must be JSON-serializable") from err
        return value


class SegmentStorePartitionConfigMismatchError(Exception):
    """Raised when preparing a partition incarnation with a different configuration than it was prepared with."""

    def __init__(
        self,
        incarnation: UUID,
        existing_config: SegmentStorePartitionConfig,
        requested_config: SegmentStorePartitionConfig,
    ) -> None:
        """Initialize with the incarnation and configurations."""
        self.incarnation = incarnation
        self.existing_config = existing_config
        self.requested_config = requested_config
        super().__init__(
            f"Partition incarnation {incarnation} was prepared with a different "
            f"configuration. Existing config: {existing_config.model_dump_json()}, "
            f"requested config: {requested_config.model_dump_json()}."
        )

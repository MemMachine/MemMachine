"""Tests for segment store data types."""

from typing import Any, cast

import pytest
from pydantic import ValidationError

from memmachine_server.common.payload_codec.payload_codec_config import (
    AESGCMPayloadCodecConfig,
    PlaintextPayloadCodecConfig,
)
from memmachine_server.episodic_memory.event_memory.segment_store import (
    SegmentStorePartitionAlreadyExistsError,
    SegmentStorePartitionConfig,
    SegmentStorePartitionConfigMismatchError,
)


def test_segment_store_partition_config_defaults_to_plaintext_codec() -> None:
    config = SegmentStorePartitionConfig()

    assert config.payload_codec_config == PlaintextPayloadCodecConfig()


def test_segment_store_partition_config_round_trip() -> None:
    config = SegmentStorePartitionConfig(
        payload_codec_config=AESGCMPayloadCodecConfig(
            key_ref="partition_key",
            wrapped_dek=b"payload-key",
            nonce_size=12,
            associated_data=b"partition:context",
        )
    )

    clone = SegmentStorePartitionConfig.model_validate(config.model_dump(mode="json"))

    assert clone == config


def test_segment_store_partition_config_rejects_non_json_serializable_values() -> None:
    with pytest.raises(ValidationError):
        SegmentStorePartitionConfig(
            payload_codec_config=cast(
                Any,
                {
                    "type": "aes_gcm",
                    "key_ref": "partition_key",
                    "wrapped_dek": object(),
                    "nonce_size": 12,
                },
            )
        )


def test_segment_store_partition_config_mismatch_error_message() -> None:
    existing = SegmentStorePartitionConfig()
    requested = SegmentStorePartitionConfig(
        payload_codec_config=AESGCMPayloadCodecConfig(
            key_ref="partition_key",
            wrapped_dek=b"payload-key",
            nonce_size=12,
            associated_data=b"partition:context",
        ),
    )

    err = SegmentStorePartitionConfigMismatchError(
        "partition_key",
        existing,
        requested,
    )

    assert err.partition_key == "partition_key"
    assert err.existing_config == existing
    assert err.requested_config == requested
    assert "already exists with a different configuration" in str(err)


def test_segment_store_partition_already_exists_error_message() -> None:
    err = SegmentStorePartitionAlreadyExistsError("partition_key")

    assert err.partition_key == "partition_key"
    assert str(err) == "Partition 'partition_key' already exists."

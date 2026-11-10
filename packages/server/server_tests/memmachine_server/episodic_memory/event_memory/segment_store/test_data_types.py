"""Tests for segment store data types."""

from uuid import UUID

from memmachine_server.common.payload_codec.payload_codec_config import (
    PlaintextPayloadCodecConfig,
)
from memmachine_server.episodic_memory.event_memory.segment_store import (
    SegmentStorePartitionConfig,
    SegmentStorePartitionConfigMismatchError,
)


def test_segment_store_partition_config_defaults_to_plaintext_codec() -> None:
    config = SegmentStorePartitionConfig()

    assert config.payload_codec_config == PlaintextPayloadCodecConfig()


def test_segment_store_partition_config_round_trip() -> None:
    config = SegmentStorePartitionConfig(
        payload_codec_config=PlaintextPayloadCodecConfig(),
    )

    clone = SegmentStorePartitionConfig.model_validate(config.model_dump(mode="json"))

    assert clone == config


def test_segment_store_partition_config_mismatch_error_names_the_incarnation() -> None:
    incarnation = UUID("00000000-0000-4000-8000-000000000001")
    config = SegmentStorePartitionConfig()

    err = SegmentStorePartitionConfigMismatchError(incarnation, config, config)

    assert err.incarnation == incarnation
    assert str(incarnation) in str(err)

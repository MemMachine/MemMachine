"""Tests for segment store data types."""

from memmachine_server.common.payload_codec.payload_codec_config import (
    PlaintextPayloadCodecConfig,
)
from memmachine_server.episodic_memory.event_memory.segment_store import (
    SegmentStorePartitionAlreadyExistsError,
    SegmentStorePartitionConfig,
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


def test_segment_store_partition_already_exists_error_message() -> None:
    err = SegmentStorePartitionAlreadyExistsError("partition_key")

    assert err.partition_key == "partition_key"
    assert str(err) == "Partition 'partition_key' already exists."

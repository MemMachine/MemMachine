"""Tests for event memory store data types."""

from memmachine_server.common.payload_codec.payload_codec_config import (
    PlaintextPayloadCodecConfig,
)
from memmachine_server.episodic_memory.event_memory.event_memory_store import (
    EventMemoryStorePartitionAlreadyExistsError,
    EventMemoryStorePartitionConfig,
)


def test_event_memory_store_partition_config_defaults_to_plaintext_codec() -> None:
    config = EventMemoryStorePartitionConfig()

    assert config.payload_codec_config == PlaintextPayloadCodecConfig()


def test_event_memory_store_partition_config_round_trip() -> None:
    config = EventMemoryStorePartitionConfig(
        payload_codec_config=PlaintextPayloadCodecConfig(),
    )

    clone = EventMemoryStorePartitionConfig.model_validate(
        config.model_dump(mode="json")
    )

    assert clone == config


def test_event_memory_store_partition_already_exists_error_message() -> None:
    err = EventMemoryStorePartitionAlreadyExistsError("partition_key")

    assert err.partition_key == "partition_key"
    assert str(err) == "Partition 'partition_key' already exists."

from datetime import timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from memmachine_server.common.configuration import SemanticMemoryConf


def test_semantic_config_with_ingestion_triggers():
    raw_conf: dict[str, Any] = {
        "database": "database",
        "llm_model": "llm",
        "embedding_model": "embedding",
        "ingestion_trigger_messages": 24,
        "ingestion_trigger_age": "PT2M",
        "config_database": "database",
    }
    conf = SemanticMemoryConf(**raw_conf)
    assert conf.ingestion_trigger_messages == 24
    assert conf.ingestion_trigger_age == timedelta(minutes=2)


def test_semantic_config_timedelta_float():
    raw_conf: dict[str, Any] = {
        "database": "database",
        "llm_model": "llm",
        "embedding_model": "embedding",
        "ingestion_trigger_messages": 24,
        "ingestion_trigger_age": 120.5,
        "config_database": "database",
    }

    conf = SemanticMemoryConf(**raw_conf)
    assert conf.ingestion_trigger_messages == 24
    assert conf.ingestion_trigger_age == timedelta(minutes=2, milliseconds=500)


def test_semantic_config_ingestion_settings_defaults_and_overrides():
    default_conf = SemanticMemoryConf(enabled=False)
    assert default_conf.ingestion_poll_interval_seconds == 2
    assert default_conf.consolidation_threshold == 20

    raw_conf: dict[str, Any] = {
        "database": "database",
        "llm_model": "llm",
        "embedding_model": "embedding",
        "config_database": "database",
        "ingestion_poll_interval_seconds": 15,
        "consolidation_threshold": 5,
    }
    conf = SemanticMemoryConf(**raw_conf)
    assert conf.ingestion_poll_interval_seconds == 15
    assert conf.consolidation_threshold == 5


@pytest.mark.parametrize("value", [0, -1])
def test_semantic_config_rejects_non_positive_poll_interval(value):
    with pytest.raises(ValidationError, match="ingestion_poll_interval_seconds"):
        SemanticMemoryConf(enabled=False, ingestion_poll_interval_seconds=value)


def test_semantic_config_accepts_zero_consolidation_threshold():
    # 0 means consolidate every tag group regardless of size.
    conf = SemanticMemoryConf(enabled=False, consolidation_threshold=0)
    assert conf.consolidation_threshold == 0


def test_semantic_config_rejects_negative_consolidation_threshold():
    with pytest.raises(ValidationError, match="consolidation_threshold"):
        SemanticMemoryConf(enabled=False, consolidation_threshold=-1)


def test_semantic_config_disabled_needs_no_config_database():
    conf = SemanticMemoryConf(enabled=False)
    assert conf.enabled is False
    assert conf.config_database is None


def test_semantic_config_auto_disables_when_config_database_missing():
    conf = SemanticMemoryConf(
        database="database",
        llm_model="llm",
        embedding_model="embedding",
    )
    assert conf.enabled is False


def test_semantic_config_stays_enabled_when_complete():
    conf = SemanticMemoryConf(
        database="database",
        config_database="database",
        llm_model="llm",
        embedding_model="embedding",
    )
    assert conf.enabled is True

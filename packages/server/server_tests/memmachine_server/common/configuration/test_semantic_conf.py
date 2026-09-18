from datetime import timedelta
from typing import Any

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

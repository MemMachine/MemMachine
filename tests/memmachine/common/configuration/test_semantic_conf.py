from datetime import timedelta
from typing import Any

from memmachine.common.configuration import SemanticMemoryConf


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


def test_semantic_config_cluster_settings():
    raw_conf: dict[str, Any] = {
        "database": "database",
        "llm_model": "llm",
        "embedding_model": "embedding",
        "ingestion_trigger_messages": 5,
        "ingestion_trigger_age": "PT1M",
        "cluster_similarity_threshold": 0.45,
        "cluster_max_time_gap": 300,
        "config_database": "database",
    }

    conf = SemanticMemoryConf(**raw_conf)

    assert conf.cluster_similarity_threshold == 0.45
    assert conf.cluster_max_time_gap == timedelta(seconds=300)

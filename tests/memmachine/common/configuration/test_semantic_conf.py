from datetime import timedelta

from memmachine.common.configuration import SemanticMemoryConf


def test_semantic_config_with_ingestion_triggers():
    raw_conf = {
        "database": "database",
        "llm_model": "llm",
        "embedding_model": "embedding",
        "ingestion_trigger_messages": 24,
        "ingestion_trigger_age": "PT2M",
    }
    conf = SemanticMemoryConf(**raw_conf)
    assert conf.ingestion_trigger_messages == 24
    assert conf.ingestion_trigger_age == timedelta(minutes=2)


def test_semantic_config_timedelta_float():
    raw_conf = {
        "database": "database",
        "llm_model": "llm",
        "embedding_model": "embedding",
        "ingestion_trigger_messages": 24,
        "ingestion_trigger_age": 120.5,
    }

    conf = SemanticMemoryConf(**raw_conf)
    assert conf.ingestion_trigger_messages == 24
    assert conf.ingestion_trigger_age == timedelta(minutes=2, milliseconds=500)

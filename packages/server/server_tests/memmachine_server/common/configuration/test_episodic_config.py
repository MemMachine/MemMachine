from typing import Any

import pytest
import yaml
from pydantic import TypeAdapter

from memmachine_server.common.configuration.episodic_config import (
    DateparserTemporalExtractorConf,
    DucklingTemporalExtractorConf,
    EpisodicMemoryConfPartial,
    LanguageModelTemporalExtractorConf,
    SegmenterConf,
    TemporalSegmenterConf,
    TextSegmenterConf,
)

_SEGMENTER_ADAPTER = TypeAdapter(SegmenterConf)


@pytest.fixture
def episodic_memory_conf() -> dict[str, Any]:
    return {
        "long_term_memory": {
            "embedder": "my_embedder",
            "reranker": "my_reranker",
            "vector_graph_store": "my_neo4j",
        },
        "short_term_memory": {
            "llm_model": "my_model",
            "message_capacity": 500,
        },
    }


def test_episodic_config_to_yaml(episodic_memory_conf):
    conf = EpisodicMemoryConfPartial(**episodic_memory_conf)
    yaml_str = conf.to_yaml()
    conf_cp = EpisodicMemoryConfPartial(**yaml.safe_load(yaml_str))
    assert conf_cp == conf
    assert conf_cp.long_term_memory == conf.long_term_memory
    assert conf_cp.short_term_memory is not None
    assert conf_cp.short_term_memory == conf.short_term_memory
    assert conf_cp.short_term_memory.llm_model == "my_model"


def test_temporal_segmenter_config_defaults_base_to_text():
    conf = _SEGMENTER_ADAPTER.validate_python(
        {"type": "temporal", "extractor": {"type": "dateparser"}}
    )
    assert isinstance(conf, TemporalSegmenterConf)
    assert isinstance(conf.extractor, DateparserTemporalExtractorConf)
    assert isinstance(conf.base_segmenter, TextSegmenterConf)


def test_temporal_segmenter_config_discriminates_extractor():
    lm = _SEGMENTER_ADAPTER.validate_python(
        {
            "type": "temporal",
            "extractor": {"type": "language_model", "language_model": "gpt"},
        }
    )
    assert isinstance(lm, TemporalSegmenterConf)
    assert isinstance(lm.extractor, LanguageModelTemporalExtractorConf)
    assert lm.extractor.language_model == "gpt"

    duckling = _SEGMENTER_ADAPTER.validate_python(
        {"type": "temporal", "extractor": {"type": "duckling", "url": "http://d/parse"}}
    )
    assert isinstance(duckling, TemporalSegmenterConf)
    assert isinstance(duckling.extractor, DucklingTemporalExtractorConf)
    assert duckling.extractor.url == "http://d/parse"


def test_temporal_segmenter_config_round_trips():
    conf = _SEGMENTER_ADAPTER.validate_python(
        {
            "type": "temporal",
            "extractor": {"type": "duckling"},
            "base_segmenter": {"type": "passthrough"},
        }
    )
    assert _SEGMENTER_ADAPTER.validate_python(conf.model_dump()) == conf

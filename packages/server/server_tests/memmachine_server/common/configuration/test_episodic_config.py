from typing import Any

import pytest
import yaml
from pydantic import TypeAdapter

from memmachine_server.common.configuration.episodic_config import (
    DateparserTemporalExtractorConf,
    DucklingTemporalExtractorConf,
    EpisodicMemoryConfPartial,
    EventLongTermMemoryConf,
    ExtractorTemporalQueryPlannerConf,
    LanguageModelTemporalExtractorConf,
    LanguageModelTemporalQueryPlannerConf,
    SegmenterConf,
    TemporalRetrievalConf,
    TemporalSegmenterConf,
    TextSegmenterConf,
)

_SEGMENTER_ADAPTER = TypeAdapter(SegmenterConf)
_TEMPORAL_RETRIEVAL_ADAPTER = TypeAdapter(TemporalRetrievalConf)


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


def test_temporal_retrieval_defaults_off_on_event_backend():
    conf = EventLongTermMemoryConf(
        session_id="s", vector_store="v", segment_store="db", embedder="e"
    )
    assert conf.temporal_retrieval is None


def test_temporal_retrieval_planner_discriminates_and_defaults():
    conf = _TEMPORAL_RETRIEVAL_ADAPTER.validate_python(
        {"planner": {"type": "language_model", "language_model": "gpt"}}
    )
    assert isinstance(conf.planner, LanguageModelTemporalQueryPlannerConf)
    assert conf.planner.language_model == "gpt"
    # k2 = one-third of k; wide overfetch; gate at >0.
    assert conf.overfetch_multiplier == 8
    assert conf.temporal_fraction == pytest.approx(1.0 / 3.0)
    assert conf.match_threshold == 0.0


def test_temporal_retrieval_extractor_planner_nests_extractor_union():
    conf = _TEMPORAL_RETRIEVAL_ADAPTER.validate_python(
        {
            "planner": {"type": "extractor", "extractor": {"type": "dateparser"}},
            "overfetch_multiplier": 12,
            "temporal_fraction": 0.5,
            "match_threshold": 0.1,
        }
    )
    assert isinstance(conf.planner, ExtractorTemporalQueryPlannerConf)
    assert isinstance(conf.planner.extractor, DateparserTemporalExtractorConf)
    assert conf.overfetch_multiplier == 12
    assert conf.temporal_fraction == 0.5
    assert conf.match_threshold == 0.1


def test_temporal_retrieval_rejects_out_of_range_fraction():
    with pytest.raises(ValueError, match="temporal_fraction"):
        _TEMPORAL_RETRIEVAL_ADAPTER.validate_python(
            {
                "planner": {"type": "language_model", "language_model": "gpt"},
                "temporal_fraction": 1.5,
            }
        )


def test_temporal_retrieval_round_trips():
    conf = _TEMPORAL_RETRIEVAL_ADAPTER.validate_python(
        {"planner": {"type": "extractor", "extractor": {"type": "duckling"}}}
    )
    assert _TEMPORAL_RETRIEVAL_ADAPTER.validate_python(conf.model_dump()) == conf

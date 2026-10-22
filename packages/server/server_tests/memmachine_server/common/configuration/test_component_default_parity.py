"""Guard intentionally duplicated configuration/component scalar defaults.

These pairs are explicit: matching field names alone do not imply equivalent
settings. Partial configurations intentionally defer with None, while injected
clients and required model IDs have no component default to compare. Settings
from unmerged implementations are not part of this manifest. This checks default
values and their types, not every validation constraint.
"""

from dataclasses import MISSING, fields, is_dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution
from importlib.util import find_spec
from operator import attrgetter

import pytest
from pydantic import BaseModel

from memmachine_server.common.configuration import (
    SemanticMemoryConf,
    SessionManagerConf,
)
from memmachine_server.common.configuration.database_conf import (
    NebulaGraphConf,
    Neo4jConf,
    SQLiteVectorStoreConf,
)
from memmachine_server.common.configuration.embedder_conf import (
    AmazonBedrockEmbedderConf,
    OpenAIEmbedderConf,
)
from memmachine_server.common.configuration.episodic_config import (
    DeclarativeLongTermMemoryConf,
    ShortTermMemoryConf,
)
from memmachine_server.common.configuration.language_model_conf import (
    AmazonBedrockLanguageModelConf,
    LiteLLMLanguageModelConf,
    OpenAIChatCompletionsLanguageModelConf,
    OpenAIResponsesLanguageModelConf,
)
from memmachine_server.common.configuration.reranker_conf import (
    BM25RerankerConf,
    CohereRerankerConf,
    RRFHybridRerankerConf,
)


def _nebula_extra_missing() -> bool:
    try:
        distribution("nebula5-python")
    except PackageNotFoundError:
        return find_spec("nebulagraph_python") is None
    # An installed but broken SDK must fail when its component is imported.
    return False


_REQUIRES_NEBULA = pytest.mark.skipif(
    _nebula_extra_missing(), reason="requires the memmachine-server nebula extra"
)

# Component imports stay lazy: the Nebula SDK is optional, and no provider or
# store needs to be constructed to inspect these declarations.
_DEFAULT_PAIRS: list[tuple[type[BaseModel], str, str]] = [
    (
        AmazonBedrockEmbedderConf,
        "common.embedder.amazon_bedrock_embedder:AmazonBedrockEmbedderParams",
        "max_retry_interval_seconds",
    ),
    (
        OpenAIEmbedderConf,
        "common.embedder.openai_embedder:OpenAIEmbedderParams",
        "max_retry_interval_seconds",
    ),
    (
        OpenAIResponsesLanguageModelConf,
        "common.language_model.openai_responses_language_model:OpenAIResponsesLanguageModelParams",
        "max_retry_interval_seconds",
    ),
    (
        OpenAIChatCompletionsLanguageModelConf,
        "common.language_model.openai_chat_completions_language_model:OpenAIChatCompletionsLanguageModelParams",
        "max_retry_interval_seconds",
    ),
    (
        AmazonBedrockLanguageModelConf,
        "common.language_model.amazon_bedrock_language_model:AmazonBedrockLanguageModelParams",
        "max_retry_interval_seconds",
    ),
    (
        LiteLLMLanguageModelConf,
        "common.language_model.litellm_language_model:LiteLLMLanguageModelParams",
        "max_retry_interval_seconds",
    ),
    (
        BM25RerankerConf,
        "common.reranker.bm25_reranker:BM25RerankerParams",
        "k1",
    ),
    (
        BM25RerankerConf,
        "common.reranker.bm25_reranker:BM25RerankerParams",
        "b",
    ),
    (
        BM25RerankerConf,
        "common.reranker.bm25_reranker:BM25RerankerParams",
        "epsilon",
    ),
    (
        RRFHybridRerankerConf,
        "common.reranker.rrf_hybrid_reranker:RRFHybridRerankerParams",
        "k",
    ),
    (
        ShortTermMemoryConf,
        "episodic_memory.short_term_memory.short_term_memory:ShortTermMemoryParams",
        "message_capacity",
    ),
    (
        SemanticMemoryConf,
        "semantic_memory.semantic_ingestion:IngestionService.Params",
        "max_features_per_update",
    ),
    (
        SemanticMemoryConf,
        "semantic_memory.semantic_memory:SemanticService.Params",
        "max_features_per_update",
    ),
    (
        SQLiteVectorStoreConf,
        "common.vector_store.sqlite_vector_store:SQLiteVectorStoreParams",
        "save_threshold",
    ),
    (
        SessionManagerConf,
        "episodic_memory.episodic_memory_manager:EpisodicMemoryManagerParams",
        "instance_cache_size",
    ),
    (
        Neo4jConf,
        "common.vector_graph_store.neo4j_vector_graph_store:Neo4jVectorGraphStoreParams",
        "force_exact_similarity_search",
    ),
    (
        LiteLLMLanguageModelConf,
        "common.language_model.litellm_language_model:LiteLLMLanguageModelParams",
        "drop_params",
    ),
    (
        DeclarativeLongTermMemoryConf,
        "episodic_memory.long_term_memory.long_term_memory:DeclarativeBackendParams",
        "message_sentence_chunking",
    ),
    (
        CohereRerankerConf,
        "common.reranker.cohere_reranker:CohereRerankerParams",
        "model",
    ),
    (
        NebulaGraphConf,
        "common.vector_graph_store.nebula_graph_vector_graph_store:NebulaGraphVectorGraphStoreParams",
        "ann_index_type",
    ),
    (
        NebulaGraphConf,
        "common.vector_graph_store.nebula_graph_vector_graph_store:NebulaGraphVectorGraphStoreParams",
        "ivf_nlist",
    ),
    (
        NebulaGraphConf,
        "common.vector_graph_store.nebula_graph_vector_graph_store:NebulaGraphVectorGraphStoreParams",
        "ivf_nprobe",
    ),
    (
        NebulaGraphConf,
        "common.vector_graph_store.nebula_graph_vector_graph_store:NebulaGraphVectorGraphStoreParams",
        "hnsw_max_degree",
    ),
    (
        NebulaGraphConf,
        "common.vector_graph_store.nebula_graph_vector_graph_store:NebulaGraphVectorGraphStoreParams",
        "hnsw_ef_construction",
    ),
    (
        NebulaGraphConf,
        "common.vector_graph_store.nebula_graph_vector_graph_store:NebulaGraphVectorGraphStoreParams",
        "hnsw_ef_search",
    ),
    (
        NebulaGraphConf,
        "common.vector_graph_store.nebula_graph_vector_graph_store:NebulaGraphVectorGraphStoreParams",
        "force_exact_similarity_search",
    ),
]


def _scalar_default(model: type, field_name: str) -> object:
    name = f"{model.__qualname__}.{field_name}"
    if issubclass(model, BaseModel):
        assert field_name in model.model_fields, f"{name} no longer exists"
        field = model.model_fields[field_name]
        assert field.default_factory is None, f"{name} now uses a default factory"
        assert not field.is_required(), f"{name} no longer has a default"
        default = field.default
    else:
        assert is_dataclass(model), f"{model.__qualname__} is not a supported model"
        declared_fields = {field.name: field for field in fields(model)}
        assert field_name in declared_fields, f"{name} no longer exists"
        dataclass_field = declared_fields[field_name]
        assert dataclass_field.default_factory is MISSING, (
            f"{name} now uses a default factory"
        )
        assert dataclass_field.default is not MISSING, f"{name} no longer has a default"
        default = dataclass_field.default
    assert type(default) in (bool, int, float, str), (
        f"{name} no longer has a scalar default: {default!r}"
    )
    return default


@pytest.mark.parametrize(
    ("config_model", "component_path", "field_name"),
    [
        pytest.param(
            config,
            component,
            field,
            id=f"{config.__name__}.{field}-{component.split(':')[1]}.{field}",
            marks=_REQUIRES_NEBULA if config is NebulaGraphConf else (),
        )
        for config, component, field in _DEFAULT_PAIRS
    ],
)
def test_component_default_parity(
    config_model: type[BaseModel], component_path: str, field_name: str
) -> None:
    module_name, class_name = component_path.split(":")
    component_model = attrgetter(class_name)(
        import_module(f"memmachine_server.{module_name}")
    )
    config_default = _scalar_default(config_model, field_name)
    component_default = _scalar_default(component_model, field_name)

    assert type(config_default) is type(component_default), (
        f"{config_model.__name__}.{field_name} has default type "
        f"{type(config_default).__name__}, but {class_name}.{field_name} has "
        f"{type(component_default).__name__}"
    )
    assert config_default == component_default, (
        f"{config_model.__name__}.{field_name} defaults to {config_default!r}, "
        f"but {class_name}.{field_name} defaults to {component_default!r}"
    )

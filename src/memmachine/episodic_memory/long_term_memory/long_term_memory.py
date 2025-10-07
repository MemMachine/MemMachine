from typing import Any

from memmachine.common.bootstrap_initializer import BootstrapInitializer

from ..data_types import ContentType, Episode, MemoryContext
from ..declarative_memory.data_types import (
    ContentType as DeclarativeMemoryContentType,
)
from ..declarative_memory.data_types import Episode as DeclarativeMemoryEpisode

supported_derivative_deriver_names = {"identity", "sentence"}

supported_reranker_names = {
    "identity",
    "bm25",
    "cross-encoder",
    "rrf-hybrid",
}

content_type_to_declarative_memory_content_type_map = {
    ContentType.STRING: DeclarativeMemoryContentType.STRING,
}

declarative_memory_content_type_to_content_type_map = {
    DeclarativeMemoryContentType.STRING: ContentType.STRING,
}


class LongTermMemory:
    def __init__(self, config: dict[str, Any], memory_context: MemoryContext):
        self._memory_context = memory_context
        long_term_memory_config = config.get("long_term_memory") or {}
        if not isinstance(long_term_memory_config, dict):
            raise TypeError("Long-term memory config must be a dictionary")

        # Configure embedder
        embedder_configs = config.get("embedder") or {}
        embedder_id = long_term_memory_config.get("embedder")
        embedder_config = embedder_configs.get(embedder_id) or {}

        embedder_model_name = embedder_config.get("model_name")
        if not isinstance(embedder_model_name, str):
            raise TypeError("Embedder model name must be provided as a string")

        embedder_api_key = embedder_config.get("api_key")
        if not isinstance(embedder_api_key, str):
            raise TypeError("Embedder API key must be provided as a string")

        # Configure vector graph store
        storage_configs = config.get("storage") or {}
        vector_graph_store_id = long_term_memory_config.get("vector_graph_store")
        vector_graph_store_config = storage_configs.get(vector_graph_store_id) or {}

        if vector_graph_store_config.get("vendor_name") != "neo4j":
            raise ValueError("Only Neo4j vector graph store is supported")

        neo4j_host = vector_graph_store_config.get("host")
        if not isinstance(neo4j_host, str):
            raise TypeError("Neo4j host must be provided as a string")

        neo4j_port = vector_graph_store_config.get("port")
        if not isinstance(neo4j_port, int):
            raise TypeError("Neo4j port must be provided as an integer")

        neo4j_uri = f"bolt://{neo4j_host}:{neo4j_port}"

        neo4j_username = vector_graph_store_config.get("user")
        if not isinstance(neo4j_username, str):
            raise TypeError("Neo4j username must be provided as a string")

        neo4j_password = vector_graph_store_config.get("password")
        if not isinstance(neo4j_password, str):
            raise TypeError("Neo4j password must be provided as a string")

        neo4j_force_exact_similarity_search = vector_graph_store_config.get(
            "force_exact_similarity_search", False
        )

        # Configure derivative deriver
        derivative_deriver_name = long_term_memory_config.get(
            "derivative_deriver", "sentence"
        )
        if derivative_deriver_name not in supported_derivative_deriver_names:
            raise ValueError(
                f"Unsupported derivative deriver name: {derivative_deriver_name}"
            )

        # Configure metadata derivative mutator
        metadata_prefix = long_term_memory_config.get(
            "metadata_prefix",
            "[$timestamp] $producer_id: ",
        )
        if not isinstance(metadata_prefix, str):
            raise TypeError("Metadata prefix must be a string")

        derivative_metadata_template = f"{metadata_prefix}$content"
        episode_metadata_template = f"{metadata_prefix}$content"

        # Configure rerankers
        reranker_configs = config.get("reranker") or {}
        if not isinstance(reranker_configs, dict):
            raise TypeError("Reranker configs must be a dictionary")

        reranker_resource_definitions = {
            reranker_id: {
                "type": "reranker",
                "name": reranker_config.get("type"),
                "config": reranker_config,
            }
            for reranker_id, reranker_config in reranker_configs.items()
        }

        reranker_id = long_term_memory_config.get("reranker")

        derivation_workflow_definition = {
            "related_episode_postulator_id": ("_null_related_episode_postulator"),
            "derivative_derivation_workflows": [
                {
                    "derivative_deriver_id": ("_episode_derivative_deriver"),
                    "derivative_mutation_workflows": [
                        {
                            "derivative_mutator_id": ("_metadata_derivative_mutator"),
                        },
                    ],
                },
            ],
        }

        resource_definitions = {
            "_declarative_memory": {
                "type": "declarative_memory",
                "name": "declarative_memory",
                "config": {
                    "vector_graph_store_id": vector_graph_store_id,
                    "embedder_id": embedder_id,
                    "reranker_id": reranker_id,
                    "query_derivative_deriver_id": "_query_derivative_deriver",
                    "related_episode_postulator_ids": [
                        "_previous_related_episode_postulator"
                    ],
                    "derivation_workflows": {
                        "default": [derivation_workflow_definition],
                    },
                    "episode_metadata_template": episode_metadata_template,
                },
            },
            "_previous_related_episode_postulator": {
                "type": "related_episode_postulator",
                "name": "previous",
                "config": {
                    "vector_graph_store_id": vector_graph_store_id,
                    "filterable_property_keys": [
                        "group_id",
                        "session_id",
                    ],
                },
            },
            "_query_derivative_deriver": {
                "type": "derivative_deriver",
                "name": "identity",
                "config": {},
            },
            "_metadata_derivative_mutator": {
                "type": "derivative_mutator",
                "name": "metadata",
                "config": {
                    "template": derivative_metadata_template,
                },
            },
            "_episode_derivative_deriver": {
                "type": "derivative_deriver",
                "name": derivative_deriver_name,
                "config": {},
            },
            "_null_related_episode_postulator": {
                "type": "related_episode_postulator",
                "name": "null",
                "config": {},
            },
            embedder_id: {
                "type": "embedder",
                "name": "openai",
                "config": {
                    "model": embedder_model_name,
                    "api_key": embedder_api_key,
                    "metrics_factory_id": "_metrics_factory",
                },
            },
            vector_graph_store_id: {
                "type": "vector_graph_store",
                "name": "neo4j",
                "config": {
                    "uri": neo4j_uri,
                    "username": neo4j_username,
                    "password": neo4j_password,
                    "force_exact_similarity_search": neo4j_force_exact_similarity_search,
                },
            },
            "_metrics_factory": {
                "type": "metrics_factory",
                "name": "prometheus",
                "config": {},
            },
        }

        resources = BootstrapInitializer.initialize(
            resource_definitions | reranker_resource_definitions
        )

        self._declarative_memory = resources.get("_declarative_memory")

    async def add_episode(self, episode: Episode):
        declarative_memory_episode = DeclarativeMemoryEpisode(
            uuid=episode.uuid,
            episode_type="default",
            content_type=content_type_to_declarative_memory_content_type_map[
                episode.content_type
            ],
            content=episode.content,
            timestamp=episode.timestamp,
            filterable_properties={
                key: value
                for key, value in {
                    "group_id": episode.group_id,
                    "session_id": episode.session_id,
                    "producer_id": episode.producer_id,
                    "produced_for_id": episode.produced_for_id,
                }.items()
                if value is not None
            },
            user_metadata=episode.user_metadata,
        )
        await self._declarative_memory.add_episode(declarative_memory_episode)

    async def search(
        self,
        query: str,
        num_episodes_limit: int,
        id_filter: dict[str, str] = {},
    ):
        declarative_memory_episodes = await self._declarative_memory.search(
            query,
            num_episodes_limit=num_episodes_limit,
            property_filter=id_filter,
        )
        return [
            Episode(
                uuid=declarative_memory_episode.uuid,
                episode_type=declarative_memory_episode.episode_type,
                content_type=(
                    declarative_memory_content_type_to_content_type_map[
                        declarative_memory_episode.content_type
                    ]
                ),
                content=declarative_memory_episode.content,
                timestamp=declarative_memory_episode.timestamp,
                group_id=declarative_memory_episode.filterable_properties.get(
                    "group_id", ""
                ),
                session_id=declarative_memory_episode.filterable_properties.get(
                    "session_id", ""
                ),
                producer_id=(
                    declarative_memory_episode.filterable_properties.get(
                        "producer_id", ""
                    )
                ),
                produced_for_id=(
                    declarative_memory_episode.filterable_properties.get(
                        "produced_for_id", ""
                    )
                ),
                user_metadata=declarative_memory_episode.user_metadata,
            )
            for declarative_memory_episode in declarative_memory_episodes
        ]

    async def clear(self):
        self._declarative_memory.forget_all()

    async def forget_session(self):
        await self._declarative_memory.forget_filtered_episodes(
            filterable_properties={
                "group_id": self._memory_context.group_id,
                "session_id": self._memory_context.session_id,
            }
        )

    async def close(self):
        await self._declarative_memory.close()

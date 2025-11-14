"""
Builder for VectorGraphStore instances.
"""

from typing import Any

from neo4j import AsyncGraphDatabase
from pydantic import BaseModel, Field, SecretStr

from memmachine.common.builder import Builder

from .vector_graph_store import VectorGraphStore


class VectorGraphStoreBuilder(Builder):
    """
    Builder for VectorGraphStore instances.
    """

    @staticmethod
    def get_dependency_ids(name: str, config: dict[str, Any]) -> set[str]:
        dependency_ids: set[str] = set()

        match name:
            case "neo4j":
                pass

        return dependency_ids

    @staticmethod
    def build(
        name: str, config: dict[str, Any], injections: dict[str, Any]
    ) -> VectorGraphStore:
        match name:
            case "neo4j":
                from .neo4j_vector_graph_store import (
                    Neo4jVectorGraphStore,
                    Neo4jVectorGraphStoreParams,
                )

                class Neo4jFactoryParams(BaseModel):
                    uri: str = Field(..., description="Neo4j connection URI")
                    username: str = Field(..., description="Neo4j username")
                    password: SecretStr = Field(..., description="Neo4j password")
                    max_concurrent_transactions: int = Field(
                        100,
                        description="Maximum number of concurrent transactions",
                        gt=0,
                    )
                    force_exact_similarity_search: bool = Field(
                        False, description="Whether to force exact similarity search"
                    )
                    filtered_similarity_search_fudge_factor: int = Field(
                        4,
                        description=(
                            "Fudge factor for filtered similarity search "
                            "because Neo4j vector index search does not "
                            "support pre-filtering or filtered search"
                        ),
                        gt=0,
                    )
                    exact_similarity_search_fallback_threshold: float = Field(
                        0.5,
                        description=(
                            "Threshold ratio of ANN search results to the search limit "
                            "below which to fall back to exact similarity search "
                            "when performing filtered similarity search"
                        ),
                        ge=0.0,
                        le=1.0,
                    )
                    range_index_hierarchies: list[list[str]] = Field(
                        default_factory=list,
                        description=(
                            "List of property name hierarchies "
                            "for which to create range indexes "
                            "applied to all nodes and edges"
                        ),
                    )
                    range_index_creation_threshold: int = Field(
                        10_000,
                        description=(
                            "Threshold number of entities "
                            "in a collection or having a relation "
                            "at which range indexes may be created"
                        ),
                    )
                    vector_index_creation_threshold: int = Field(
                        10_000,
                        description=(
                            "Threshold number of entities "
                            "in a collection or having a relation "
                            "at which vector indexes may be created"
                        ),
                    )

                factory_params = Neo4jFactoryParams(**config)
                driver = AsyncGraphDatabase.driver(
                    factory_params.uri,
                    auth=(
                        factory_params.username,
                        factory_params.password.get_secret_value(),
                    ),
                )

                return Neo4jVectorGraphStore(
                    Neo4jVectorGraphStoreParams(
                        driver=driver,
                        max_concurrent_transactions=factory_params.max_concurrent_transactions,
                        force_exact_similarity_search=factory_params.force_exact_similarity_search,
                        filtered_similarity_search_fudge_factor=factory_params.filtered_similarity_search_fudge_factor,
                        exact_similarity_search_fallback_threshold=factory_params.exact_similarity_search_fallback_threshold,
                        range_index_hierarchies=[
                            [
                                "filterable_group_id",
                                "filterable_session_id",
                                "timestamp",
                            ],
                        ],
                        range_index_creation_threshold=factory_params.range_index_creation_threshold,
                        vector_index_creation_threshold=factory_params.vector_index_creation_threshold,
                    )
                )
            case _:
                raise ValueError(f"Unknown VectorGraphStore name: {name}")

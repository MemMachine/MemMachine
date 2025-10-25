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
                    )
                )
            case _:
                raise ValueError(f"Unknown VectorGraphStore name: {name}")

"""
Builder for VectorGraphStore instances.
"""

from typing import Any

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
                    Neo4jVectorGraphStoreConfig,
                )

                return Neo4jVectorGraphStore(Neo4jVectorGraphStoreConfig(**config))
            case _:
                raise ValueError(f"Unknown VectorGraphStore name: {name}")

"""
Resource initializer for building resources
based on their definitions and dependencies.
"""

from collections import deque
from typing import Any

from memmachine.common.builder import Builder
from memmachine.common.embedder.embedder_builder import EmbedderBuilder
from memmachine.common.language_model.language_model_builder import (
    LanguageModelBuilder,
)
from memmachine.common.metrics_factory.metrics_factory_builder import (
    MetricsFactoryBuilder,
)
from memmachine.common.reranker.reranker_builder import RerankerBuilder
from memmachine.common.vector_graph_store.vector_graph_store_builder import (
    VectorGraphStoreBuilder,
)
from memmachine.episodic_memory.declarative_memory import (
    DeclarativeMemoryBuilder,
)
from memmachine.episodic_memory.declarative_memory.derivative_deriver import (
    DerivativeDeriverBuilder,
)
from memmachine.episodic_memory.declarative_memory.derivative_mutator import (
    DerivativeMutatorBuilder,
)
from memmachine.episodic_memory.declarative_memory.related_episode_postulator import (
    RelatedEpisodePostulatorBuilder,
)

"""
Each entry in resource_definitions should look like this:
```
resource_id: {
    "type": "<TYPE>",
    "name": "<NAME>",
    "config": {
        ... <CONFIGURATION> ...
    }
}
```
"""

# Map resource types to their corresponding builder classes
resource_builder_map: dict[str, type[Builder]] = {
    "declarative_memory": DeclarativeMemoryBuilder,
    "derivative_deriver": DerivativeDeriverBuilder,
    "derivative_mutator": DerivativeMutatorBuilder,
    "related_episode_postulator": RelatedEpisodePostulatorBuilder,
    "embedder": EmbedderBuilder,
    "language_model": LanguageModelBuilder,
    "metrics_factory": MetricsFactoryBuilder,
    "reranker": RerankerBuilder,
    "vector_graph_store": VectorGraphStoreBuilder,
}


class ResourceInitializer:
    """
    Resource initializer for building resources
    based on their definitions and dependencies.
    """

    @staticmethod
    def initialize(
        resource_definitions: dict[str, Any],
        resource_cache: dict[str, Any] | None = None,
    ):
        """
        Initialize resources
        based on their definitions and dependencies.
        """
        if resource_cache is None:
            resource_cache = {}

        resource_dependency_graph = {}

        for (
            resource_id,
            resource_definition,
        ) in resource_definitions.items():
            resource_builder = resource_builder_map[resource_definition["type"]]
            resource_dependency_graph[resource_id] = (
                resource_builder.get_dependency_ids(
                    resource_definition["name"],
                    resource_definition["config"],
                )
            )

        def order_resources(
            resource_dependency_graph: dict[str, set[str]],
        ) -> list[str]:
            """
            Order resources based on their dependencies
            using a topological sort.
            """
            ordered_resource_ids = []

            dependency_counts = {
                resource_id: 0 for resource_id in resource_dependency_graph.keys()
            }
            dependent_resource_ids: dict[str, set[str]] = {
                resource_id: set() for resource_id in resource_dependency_graph.keys()
            }

            for (
                resource_id,
                dependency_ids,
            ) in resource_dependency_graph.items():
                for dependency_id in dependency_ids:
                    # Check that the dependency exists in either the resource definitions or the resource cache.
                    if (
                        dependency_id not in resource_dependency_graph.keys()
                        and dependency_id not in resource_cache.keys()
                    ):
                        raise ValueError(
                            f"Dependency {dependency_id} "
                            f"for resource {resource_id} "
                            "found in neither resource definitions not resource cache"
                        )

                    # Only count depdencies that have not been initialized yet.
                    if dependency_id in resource_dependency_graph.keys():
                        dependency_counts[resource_id] += 1
                        dependent_resource_ids[dependency_id].add(resource_id)

            queue = deque(
                [
                    resource_id
                    for resource_id, count in dependency_counts.items()
                    if count == 0
                ]
            )

            while queue:
                resource_id = queue.popleft()
                ordered_resource_ids.append(resource_id)

                for dependent_resource_id in dependent_resource_ids[resource_id]:
                    dependency_counts[dependent_resource_id] -= 1
                    if dependency_counts[dependent_resource_id] == 0:
                        queue.append(dependent_resource_id)

            if len(ordered_resource_ids) != len(resource_dependency_graph):
                raise ValueError("Cyclic dependency detected in resource definitions")

            return ordered_resource_ids

        ordered_resource_ids = order_resources(resource_dependency_graph)

        initialized_resources: dict[str, Any] = {}
        for resource_id in ordered_resource_ids:
            if resource_id in resource_cache:
                continue

            resource_definition = resource_definitions[resource_id]

            resource_builder = resource_builder_map[resource_definition["type"]]

            initialized_resource = resource_builder.build(
                resource_definition["name"],
                resource_definition["config"],
                injections=resource_cache | initialized_resources,
            )

            initialized_resources[resource_id] = initialized_resource

        return initialized_resources

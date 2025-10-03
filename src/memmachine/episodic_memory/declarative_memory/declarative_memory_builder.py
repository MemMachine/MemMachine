"""
Builder for DeclarativeMemory instances.
"""

from typing import Any

from memmachine.common.builder import Builder

from .declarative_memory import DeclarativeMemory


class DeclarativeMemoryBuilder(Builder):
    """
    Builder for DeclarativeMemory instances.
    """

    @staticmethod
    def get_dependency_ids(name: str, config: dict[str, Any]) -> set[str]:
        dependency_ids = set()
        dependency_ids.add(config["vector_graph_store_id"])
        dependency_ids.add(config["embedder_id"])
        dependency_ids.add(config["reranker_id"])
        dependency_ids.add(config["query_derivative_deriver_id"])

        for episode_cluster_assembly_workflow_configs in config[
            "derivation_workflows"
        ].values():
            for (
                episode_cluster_assembly_workflow_config
            ) in episode_cluster_assembly_workflow_configs:
                dependency_ids.add(
                    episode_cluster_assembly_workflow_config[
                        "related_episode_postulator_id"
                    ]
                )
                for (
                    derivative_derivation_workflow_config
                ) in episode_cluster_assembly_workflow_config[
                    "derivative_derivation_workflows"
                ]:
                    dependency_ids.add(
                        derivative_derivation_workflow_config["derivative_deriver_id"]
                    )
                    for (
                        derivative_mutation_workflow_config
                    ) in derivative_derivation_workflow_config[
                        "derivative_mutation_workflows"
                    ]:
                        dependency_ids.add(
                            derivative_mutation_workflow_config["derivative_mutator_id"]
                        )

        return dependency_ids

    @staticmethod
    def build(
        name: str, config: dict[str, Any], injections: dict[str, Any]
    ) -> DeclarativeMemory:
        def build_injected_derivative_mutation_workflow_config(
            config: dict[str, Any], injections: dict[str, Any]
        ):
            return {"derivative_mutator": injections[config["derivative_mutator_id"]]}

        def build_injected_derivative_derivation_workflow_config(
            config: dict[str, Any], injections: dict[str, Any]
        ):
            return {
                "derivative_deriver": injections[config["derivative_deriver_id"]],
                "derivative_mutation_workflows": [
                    build_injected_derivative_mutation_workflow_config(
                        derivative_mutation_workflow_config, injections
                    )
                    for derivative_mutation_workflow_config in config[
                        "derivative_mutation_workflows"
                    ]
                ],
            }

        def build_injected_episode_cluster_assembler_workflow_config(
            config: dict[str, Any], injections: dict[str, Any]
        ):
            return {
                "related_episode_postulator": injections[
                    config["related_episode_postulator_id"]
                ],
                "derivative_derivation_workflows": [
                    build_injected_derivative_derivation_workflow_config(
                        derivative_derivation_workflow_config,
                        injections,
                    )
                    for derivative_derivation_workflow_config in config[
                        "derivative_derivation_workflows"
                    ]
                ],
            }

        populated_config = {
            "vector_graph_store": injections[config["vector_graph_store_id"]],
            "embedder": injections[config["embedder_id"]],
            "reranker": injections[config["reranker_id"]],
            "related_episode_postulators": [
                injections[related_episode_postulator_id]
                for related_episode_postulator_id in config.get(
                    "related_episode_postulator_ids"
                )
                or []
            ],
            "query_derivative_deriver": injections[
                config["query_derivative_deriver_id"]
            ],
            "derivation_workflows": {
                episode_type: [
                    build_injected_episode_cluster_assembler_workflow_config(
                        workflow_config, injections
                    )
                    for workflow_config in derivation_workflows
                ]
                for episode_type, derivation_workflows in config[
                    "derivation_workflows"
                ].items()
            },
            "episode_metadata_template": config["episode_metadata_template"],
        }

        return DeclarativeMemory(populated_config)

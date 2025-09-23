"""
Builder for LongTermMemory instances.
"""

from typing import Any

from memmachine.common.builder import Builder

from .long_term_memory import LongTermMemory


class LongTermMemoryBuilder(Builder):
    """
    Builder for LongTermMemory instances.
    """

    @staticmethod
    def get_dependency_ids(name: str, config: dict[str, Any]) -> set[str]:
        dependency_ids = set()
        dependency_ids.add(config["declarative_memory_id"])
        return dependency_ids

    @staticmethod
    def build(
        name: str, config: dict[str, Any], injections: dict[str, Any]
    ) -> LongTermMemory:
        return LongTermMemory(
            {
                "declarative_memory": injections[
                    config["declarative_memory_id"]
                ],
            }
        )

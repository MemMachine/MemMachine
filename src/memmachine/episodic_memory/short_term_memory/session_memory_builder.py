"""
Builder for SessionMemory instances.
"""

from typing import Any

from memmachine.common.builder import Builder

from .session_memory import SessionMemory


class SessionMemoryBuilder(Builder):
    """
    Builder for SessionMemory instances.
    """

    @staticmethod
    def get_dependency_ids(name: str, config: dict[str, Any]) -> set[str]:
        dependency_ids = set()
        dependency_ids.add(config["model_id"])
        return dependency_ids

    @staticmethod
    def build(
        name: str, config: dict[str, Any], injections: dict[str, Any]
    ) -> SessionMemory:
        return SessionMemory(
            {key: value for key, value in config.items() if key != "model_id"}
            | {
                "model": injections[config["model_id"]],
            }
        )

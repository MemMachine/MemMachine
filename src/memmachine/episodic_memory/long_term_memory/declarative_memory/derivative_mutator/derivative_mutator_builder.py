"""
Builder for DerivativeMutator instances.
"""

from typing import Any

from memmachine.common.builder import Builder

from .derivative_mutator import DerivativeMutator


class DerivativeMutatorBuilder(Builder):
    """
    Builder for DerivativeMutator instances.
    """

    @staticmethod
    def get_dependency_ids(name: str, config: dict[str, Any]) -> set[str]:
        dependency_ids = set()

        match name:
            case "identity" | "metadata":
                pass
            case "third-person-rewrite":
                dependency_ids.add(config["language_model_id"])

        return dependency_ids

    @staticmethod
    def build(
        name: str, config: dict[str, Any], injections: dict[str, Any]
    ) -> DerivativeMutator:
        match name:
            case "identity":
                from .identity_derivative_mutator import (
                    IdentityDerivativeMutator,
                )

                return IdentityDerivativeMutator()
            case "metadata":
                from .metadata_derivative_mutator import (
                    MetadataDerivativeMutator,
                )

                populated_config = config
                return MetadataDerivativeMutator(populated_config)
            case "third-person-rewrite":
                from .third_person_rewrite_derivative_mutator import (
                    ThirdPersonRewriteDerivativeMutator,
                )

                populated_config = {
                    "language_model": injections[config["language_model_id"]],
                }
                return ThirdPersonRewriteDerivativeMutator(populated_config)
            case _:
                raise ValueError(f"Unknown DerivativeMutator name: {name}")

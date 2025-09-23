"""
Builder for DerivativeDeriver instances.
"""

from typing import Any

from memmachine.common.builder import Builder

from .derivative_deriver import DerivativeDeriver


class DerivativeDeriverBuilder(Builder):
    """
    Builder for DerivativeDeriver instances.
    """

    @staticmethod
    def get_dependency_ids(name: str, config: dict[str, Any]) -> set[str]:
        dependency_ids: set[str] = set()

        match name:
            case "concatenation" | "identity" | "sentence":
                pass

        return dependency_ids

    @staticmethod
    def build(
        name: str, config: dict[str, Any], injections: dict[str, Any]
    ) -> DerivativeDeriver:
        match name:
            case "concatenation":
                from .concatenation_derivative_deriver import (
                    ConcatenationDerivativeDeriver,
                )

                populated_config = config
                return ConcatenationDerivativeDeriver(populated_config)
            case "identity":
                from .identity_derivative_deriver import (
                    IdentityDerivativeDeriver,
                )

                populated_config = config
                return IdentityDerivativeDeriver(populated_config)
            case "sentence":
                from .sentence_derivative_deriver import (
                    SentenceDerivativeDeriver,
                )

                populated_config = config
                return SentenceDerivativeDeriver(populated_config)
            case _:
                raise ValueError(f"Unknown DerivativeDeriver name: {name}")

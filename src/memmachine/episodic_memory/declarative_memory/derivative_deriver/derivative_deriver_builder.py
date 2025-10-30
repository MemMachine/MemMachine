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
                    ConcatenationDerivativeDeriverParams,
                )

                concatenation_params = ConcatenationDerivativeDeriverParams(**config)
                return ConcatenationDerivativeDeriver(concatenation_params)
            case "identity":
                from .identity_derivative_deriver import (
                    IdentityDerivativeDeriver,
                    IdentityDerivativeDeriverParams,
                )

                identity_params = IdentityDerivativeDeriverParams(**config)
                return IdentityDerivativeDeriver(identity_params)
            case "sentence":
                from .sentence_derivative_deriver import (
                    SentenceDerivativeDeriver,
                    SentenceDerivativeDeriverParams,
                )

                sentence_params = SentenceDerivativeDeriverParams(**config)
                return SentenceDerivativeDeriver(sentence_params)
            case _:
                raise ValueError(f"Unknown DerivativeDeriver name: {name}")

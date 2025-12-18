"""Metrics configuration mixins."""

from datetime import timedelta
from enum import Enum
from typing import ClassVar

import yaml
from pydantic import BaseModel, Field, SecretStr

from memmachine.common.metrics_factory import MetricsFactory
from memmachine.common.metrics_factory.prometheus_metrics_factory import (
    PrometheusMetricsFactory,
)


class UnknownMetricsFactoryError(ValueError):
    """Raised when the metrics factory name is invalid."""


class MetricsFactoryIdMixin(BaseModel):
    """Mixin for configurations that include a metrics factory ID."""

    metrics_factory_id: str | None = Field(
        default=None,
        description="Metrics factory ID for monitoring and metrics collection.",
    )
    user_metrics_labels: dict[str, str] = Field(
        default_factory=dict,
        description="User-defined labels for metrics.",
    )

    _factories: ClassVar[dict[str, MetricsFactory]] = {}

    def get_metrics_factory(self) -> MetricsFactory | None:
        """Return the configured metrics factory instance, if any."""
        factory_id = self.metrics_factory_id
        if factory_id is None:
            factory_id = "prometheus"
        if factory_id not in self._factories:
            match factory_id:
                case "prometheus":
                    factory = PrometheusMetricsFactory()
                    self._factories[factory_id] = factory
                case _:
                    raise UnknownMetricsFactoryError(
                        f"Unknown MetricsFactory name: {factory_id}"
                    )
        return self._factories[factory_id]


class YamlSerializableMixin(BaseModel):
    """Mixin that adds YAML-safe serialization for Pydantic models."""

    def to_yaml_dict(self) -> dict:
        raw = self.model_dump()

        def unwrap(obj: YamlObjType) -> YamlObjType:
            """Recursively unwrap Pydantic models, SecretStr, Enums, and drop empty values."""
            if isinstance(obj, YamlSerializableMixin):
                obj = obj.to_yaml_dict()

            # Unwrap SecretStr
            if isinstance(obj, SecretStr):
                obj = obj.get_secret_value()

            # Unwrap enums like SimilarityMetric
            if isinstance(obj, Enum):
                obj = obj.value

            if isinstance(obj, timedelta):
                obj = obj.total_seconds()

            # Dict — recurse & drop empty
            if isinstance(obj, dict):
                cleaned = {k: unwrap(v) for k, v in obj.items()}
                # drop keys whose values are None/empty
                cleaned = {
                    k: v for k, v in cleaned.items() if v not in (None, "", [], {})
                }
                return cleaned

            # List — recurse & drop empty
            if isinstance(obj, list):
                cleaned = [unwrap(v) for v in obj]
                cleaned = [v for v in cleaned if v not in (None, "", [], {})]
                return cleaned

            # Base condition
            return obj

        ret = unwrap(raw)
        if not isinstance(ret, dict):
            raise TypeError(
                "to_yaml_dict can only be called on models that serialize to dicts"
            )
        return ret

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_yaml_dict(), sort_keys=False)


type YamlObjType = (
    YamlSerializableMixin
    | SecretStr
    | Enum
    | dict[str, "YamlObjType"]
    | list["YamlObjType"]
    | str
    | int
    | float
    | bool
    | None
)

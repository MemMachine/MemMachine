"""Metrics configuration mixins."""

from typing import ClassVar

from pydantic import BaseModel, Field

from memmachine.common.metrics_factory import MetricsFactory
from memmachine.common.metrics_factory.prometheus_metrics_factory import (
    PrometheusMetricsFactory,
)


class UnknownMetricsFactoryError(ValueError):
    """Raised when the metrics factory name is invalid."""


class WithMetricsFactoryId(BaseModel):
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

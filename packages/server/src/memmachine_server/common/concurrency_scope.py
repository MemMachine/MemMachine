"""Concurrency boundaries supported by components and deployments."""

from enum import StrEnum

from memmachine_server.common.errors import ConfigurationError


class ConcurrencyScope(StrEnum):
    """The widest deployment boundary within which state can be shared."""

    PROCESS = "process"
    HOST = "host"
    CLUSTER = "cluster"


_SCOPE_ORDER = {
    ConcurrencyScope.PROCESS: 0,
    ConcurrencyScope.HOST: 1,
    ConcurrencyScope.CLUSTER: 2,
}


def validate_component_scope(
    component_name: str,
    component_scope: ConcurrencyScope,
    deployment_scope: ConcurrencyScope,
) -> None:
    """Reject a component narrower than its deployment boundary."""
    if _SCOPE_ORDER[component_scope] < _SCOPE_ORDER[deployment_scope]:
        raise ConfigurationError(
            f"{component_name} supports {component_scope.value} concurrency, but "
            f"the deployment requires {deployment_scope.value}; disable the "
            "component or use a deployment with a narrower concurrency scope."
        )

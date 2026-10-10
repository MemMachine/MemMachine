import pytest

from memmachine_server.common.concurrency_scope import (
    ConcurrencyScope,
    validate_component_scope,
)
from memmachine_server.common.errors import ConfigurationError


@pytest.mark.parametrize("component", list(ConcurrencyScope))
@pytest.mark.parametrize("deployment", list(ConcurrencyScope))
def test_scope_compatibility(component: ConcurrencyScope, deployment: ConcurrencyScope):
    scopes = list(ConcurrencyScope)
    if scopes.index(component) < scopes.index(deployment):
        with pytest.raises(
            ConfigurationError, match=f"test component supports {component}"
        ):
            validate_component_scope("test component", component, deployment)
    else:
        validate_component_scope("test component", component, deployment)

"""version information."""

import contextlib
from importlib.metadata import PackageNotFoundError, version

from memmachine.common.api.spec import Version


def _get_package_version(package_name: str) -> str:
    ret = "not available"
    with contextlib.suppress(PackageNotFoundError):
        ret = version(package_name)
    return ret


def get_version() -> Version:
    """Get the server and client version information."""
    return Version(
        server_version=_get_package_version("memmachine-server"),
        client_version=_get_package_version("memmachine-client"),
    )

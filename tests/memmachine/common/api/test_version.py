"""test version information."""

import re

from memmachine.common.api.spec import Version
from memmachine.common.api.version import get_version


def test_get_version():
    # Matches:
    # 0.2.2
    # 0.2.2.dev93
    # 0.2.2.dev93+g2b5fd8250
    pattern = re.compile(
        r"""
        ^\d+\.\d+\.\d+           # major.minor.patch
        (?:\.dev\d+(?:\+[a-z0-9]+)?)?   # optional .devN[+local]
        $
        """,
        re.VERBOSE,
    )

    version = get_version()
    assert pattern.match(version.server_version), (
        f"Invalid server version: {version.server_version}"
    )
    assert pattern.match(version.client_version), (
        f"Invalid client version: {version.client_version}"
    )


def test_version_string():
    version = Version(server_version="0.2.2", client_version="0.2.2")
    assert str(version) == "server: 0.2.2\nclient: 0.2.2"

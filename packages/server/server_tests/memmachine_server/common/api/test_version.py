"""test version information."""

from memmachine_common.api.spec import Version


def test_version_string():
    version = Version(server_version="0.2.2", client_version="0.2.2")
    assert str(version) == "server: 0.2.2\nclient: 0.2.2"

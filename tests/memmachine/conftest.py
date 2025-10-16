# conftest.py
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--manual-integration",
        action="store_true",
        default=False,
        help="Run manual integration tests",
    )


def pytest_collection_modifyitems(config, items):
    skip_integration = pytest.mark.skip(
        reason="need --manual-integration option to run"
    )

    if not config.getoption("--manual-integration"):
        for item in items:
            if "manual_integration" in item.keywords:
                item.add_marker(skip_integration)

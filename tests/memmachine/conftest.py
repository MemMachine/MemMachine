# conftest.py
from unittest.mock import create_autospec

import pytest

from memmachine.common.language_model import LanguageModel
from tests.memmachine.common.reranker.fake_embedder import FakeEmbedder


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests",
    )


def pytest_collection_modifyitems(config, items):
    skip_integration = pytest.mark.skip(reason="need --integration option to run")

    if not config.getoption("--integration"):
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

@pytest.fixture
def mock_llm_model():
    return create_autospec(LanguageModel, instance=True)

@pytest.fixture
def mock_llm_embedder():
    return FakeEmbedder()

import pytest

from memmachine.semantic_memory.semantic_ingestion import SemanticFeature


@pytest.fixture
def duplicate_features():
    return [
        SemanticFeature(
            tag="food",
            feature="favorite_pizza",
            value="peperoni pizza",
        ),
        SemanticFeature(
            tag="food",
            feature="favorite_food",
            value="peperoni pizza",
        ),
    ]


@pytest.fixture
def llm_model(mock_llm_model):
    return mock_llm_model

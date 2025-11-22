import pytest

from memmachine.semantic_memory.semantic_llm import (
    llm_consolidate_features,
    llm_feature_update,
)
from memmachine.semantic_memory.semantic_model import SemanticFeature
from memmachine.server.prompt.profile_prompt import UserProfileSemanticCategory

pytestmark = pytest.mark.integration


@pytest.fixture
def llm_model(real_llm_model):
    return real_llm_model


@pytest.fixture
def semantic_prompt():
    return UserProfileSemanticCategory.prompt


@pytest.fixture
def update_prompt(semantic_prompt):
    return semantic_prompt.update_prompt


@pytest.fixture
def consolidation_prompt(semantic_prompt):
    return semantic_prompt.consolidation_prompt


@pytest.mark.asyncio
async def test_semantic_llm_update_with_empty_profile(
    real_llm_model,
    update_prompt,
):
    commands = await llm_feature_update(
        features=[],
        message_content="I like blue cars made in Berlin, Germany",
        model=real_llm_model,
        update_prompt=update_prompt,
    )

    assert commands is not None


@pytest.mark.asyncio
async def test_semantic_llm_consolidate_with_basic_profile(
    real_llm_model,
    consolidation_prompt,
):
    result = await llm_consolidate_features(
        features=[
            SemanticFeature(
                feature_name="favorite_pizza",
                value="pepperoni",
                tag="food",
                category="Profile",
            ),
        ],
        consolidate_prompt=consolidation_prompt,
        model=real_llm_model,
    )

    assert result is not None

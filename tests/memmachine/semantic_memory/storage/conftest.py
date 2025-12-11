import numpy as np
import pytest_asyncio

from memmachine.semantic_memory.storage.storage_base import SemanticStorage


@pytest_asyncio.fixture
async def with_multiple_features(semantic_storage: SemanticStorage):
    idx_a = await semantic_storage.add_feature(
        set_id="user",
        category_name="test_type",
        feature="likes",
        value="pizza",
        tag="food",
        embedding=np.array([1.0] * 1536, dtype=float),
    )
    idx_b = await semantic_storage.add_feature(
        set_id="user",
        category_name="test_type",
        feature="likes",
        value="sushi",
        tag="food",
        embedding=np.array([1.0] * 1536, dtype=float),
    )
    key = (
        "test_type",
        "food",
        "likes",
    )
    yield (
        key,
        {
            key: [
                {
                    "value": "pizza",
                },
                {
                    "value": "sushi",
                },
            ],
        },
    )

    await semantic_storage.delete_features([idx_a, idx_b])


@pytest_asyncio.fixture
async def with_multiple_sets(semantic_storage: SemanticStorage):
    idx_a = await semantic_storage.add_feature(
        set_id="user1",
        category_name="default",
        feature="likes",
        value="pizza",
        tag="food",
        embedding=np.array([1.0] * 1536, dtype=float),
    )
    idx_b = await semantic_storage.add_feature(
        set_id="user1",
        category_name="default",
        feature="likes",
        value="sushi",
        tag="food",
        embedding=np.array([1.0] * 1536, dtype=float),
    )
    idx_c = await semantic_storage.add_feature(
        set_id="user2",
        category_name="default",
        feature="likes",
        value="fish",
        tag="food",
        embedding=np.array([1.0] * 1536, dtype=float),
    )
    idx_d = await semantic_storage.add_feature(
        set_id="user2",
        category_name="default",
        feature="likes",
        value="chips",
        tag="food",
        embedding=np.array([1.0] * 1536, dtype=float),
    )

    key = (
        "default",
        "food",
        "likes",
    )

    yield (
        key,
        {
            "user1": [
                {
                    "value": "pizza",
                },
                {
                    "value": "sushi",
                },
            ],
            "user2": [
                {
                    "value": "fish",
                },
                {
                    "value": "chips",
                },
            ],
        },
    )

    await semantic_storage.delete_features([idx_a, idx_b, idx_c, idx_d])

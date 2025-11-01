import numpy as np
import pytest_asyncio

from memmachine.semantic_memory.storage.storage_base import SemanticStorageBase


@pytest_asyncio.fixture
async def with_multiple_features(storage: SemanticStorageBase):
    idx_a = await storage.add_feature(
        set_id="user",
        type_name="test_type",
        feature="likes",
        value="pizza",
        tag="food",
        embedding=np.array([1.0] * 1536, dtype=float),
    )
    idx_b = await storage.add_feature(
        set_id="user",
        type_name="test_type",
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
            ]
        },
    )

    await storage.delete_features([idx_a, idx_b])


@pytest_asyncio.fixture
async def with_multiple_sets(storage: SemanticStorageBase):
    idx_a = await storage.add_feature(
        set_id="user1",
        type_name="default",
        feature="likes",
        value="pizza",
        tag="food",
        embedding=np.array([1.0] * 1536, dtype=float),
    )
    idx_b = await storage.add_feature(
        set_id="user1",
        type_name="default",
        feature="likes",
        value="sushi",
        tag="food",
        embedding=np.array([1.0] * 1536, dtype=float),
    )
    idx_c = await storage.add_feature(
        set_id="user2",
        type_name="default",
        feature="likes",
        value="fish",
        tag="food",
        embedding=np.array([1.0] * 1536, dtype=float),
    )
    idx_d = await storage.add_feature(
        set_id="user2",
        type_name="default",
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

    await storage.delete_features([idx_a, idx_b, idx_c, idx_d])

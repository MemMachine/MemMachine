import numpy as np
import pytest_asyncio

from memmachine.semantic_memory.storage.storage_base import SemanticStorageBase


@pytest_asyncio.fixture
async def with_multiple_features(storage: SemanticStorageBase):
    idx_a = await storage.add_feature(
        set_id="user",
        semantic_type_id="default",
        feature="likes",
        value="pizza",
        tag="food",
        embedding=np.array([1.0] * 1536, dtype=float),
    )
    idx_b = await storage.add_feature(
        set_id="user",
        semantic_type_id="default",
        feature="likes",
        value="sushi",
        tag="food",
        embedding=np.array([1.0] * 1536, dtype=float),
    )

    yield {
        "food": {
            "likes": [
                {
                    "value": "pizza",
                },
                {
                    "value": "sushi",
                },
            ]
        }
    }

    await storage.delete_features([idx_a, idx_b])


@pytest_asyncio.fixture
async def with_multiple_sets(storage: SemanticStorageBase):
    idx_a = await storage.add_feature(
        set_id="user1",
        semantic_type_id="default",
        feature="likes",
        value="pizza",
        tag="food",
        embedding=np.array([1.0] * 1536, dtype=float),
    )
    idx_b = await storage.add_feature(
        set_id="user1",
        semantic_type_id="default",
        feature="likes",
        value="sushi",
        tag="food",
        embedding=np.array([1.0] * 1536, dtype=float),
    )
    idx_c = await storage.add_feature(
        set_id="user2",
        semantic_type_id="default",
        feature="likes",
        value="fish",
        tag="food",
        embedding=np.array([1.0] * 1536, dtype=float),
    )
    idx_d = await storage.add_feature(
        set_id="user2",
        semantic_type_id="default",
        feature="likes",
        value="chips",
        tag="food",
        embedding=np.array([1.0] * 1536, dtype=float),
    )

    yield {
        "user1": {
            "food": {
                "likes": [
                    {
                        "value": "pizza",
                    },
                    {
                        "value": "sushi",
                    },
                ]
            }
        },
        "user2": {
            "food": {
                "likes": [
                    {
                        "value": "fish",
                    },
                    {
                        "value": "chips",
                    },
                ]
            }
        },
    }

    await storage.delete_features([idx_a, idx_b, idx_c, idx_d])

import numpy as np
import pytest

from memmachine.semantic_memory.storage.storage_base import SemanticStorage


@pytest.mark.asyncio
async def test_delete_history_removes_history_and_citations(
    semantic_storage: SemanticStorage,
):
    history_id = "history-1"
    await semantic_storage.add_history_to_set(
        set_id="user-a",
        history_id=history_id,
    )

    feature_id = await semantic_storage.add_feature(
        set_id="user-a",
        category_name="profile",
        feature="favorite_food",
        value="pizza",
        tag="food",
        embedding=np.array([1.0, 1.0]),
    )
    await semantic_storage.add_citations(feature_id, [history_id])

    before_history = await semantic_storage.get_history_messages(set_ids=["user-a"])
    assert history_id in before_history

    before_feature = await semantic_storage.get_feature(feature_id, load_citations=True)
    assert before_feature is not None
    before_citations = before_feature.metadata.citations or []
    assert history_id in before_citations

    await semantic_storage.delete_history([history_id])

    remaining = await semantic_storage.get_history_messages(set_ids=["user-a"])
    assert remaining == []

    feature = await semantic_storage.get_feature(feature_id, load_citations=True)
    assert feature is not None
    citations = feature.metadata.citations or []
    assert history_id not in citations

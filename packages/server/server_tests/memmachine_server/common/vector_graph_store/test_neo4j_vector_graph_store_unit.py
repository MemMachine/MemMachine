from unittest.mock import AsyncMock

import pytest
from neo4j import AsyncDriver

from memmachine_server.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
)


@pytest.mark.asyncio
async def test_populate_index_state_cache_does_not_wait_for_all_indexes():
    driver = AsyncMock(spec=AsyncDriver)
    driver.execute_query = AsyncMock(
        return_value=(
            [
                {"name": "idx_online", "state": "ONLINE"},
                {"name": "idx_populating", "state": "POPULATING"},
            ],
            None,
            None,
        )
    )
    store = Neo4jVectorGraphStore(Neo4jVectorGraphStoreParams(driver=driver))

    await store._populate_index_state_cache()

    assert store._index_state_cache == {
        "idx_online": Neo4jVectorGraphStore.CacheIndexState.ONLINE,
        "idx_populating": Neo4jVectorGraphStore.CacheIndexState.CREATING,
    }
    executed_queries = [
        call.args[0].text for call in driver.execute_query.await_args_list
    ]
    assert executed_queries == ["SHOW INDEXES YIELD name, state RETURN name, state"]

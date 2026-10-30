"""
The schema check every vector store makes when it starts and after it migrates.

Each store's test module mixes `StoreSchemaCheckContract` into a test class,
supplies the `store` fixture (a migrated and started store of
`VECTOR_STORE_NAME` declaring 3 dimensions by cosine and
`INDEXED_PROPERTIES`), and implements `build`: a store over the fixture's
backend, neither migrated nor started, of a given name and of the fixture's
schema changed by the given fields.
"""

from datetime import datetime
from typing import Any

import pytest

from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.errors import SchemaMismatchError
from memmachine_server.common.vector_store import VectorStore

_OTHER_SCHEMAS = [
    pytest.param({"vector_dimensions": 4}, id="other-dimensions"),
    pytest.param(
        {"similarity_metric": SimilarityMetric.EUCLIDEAN}, id="another-metric"
    ),
    pytest.param({"indexed_properties": {"name": str}}, id="fewer-properties"),
    pytest.param(
        {
            "indexed_properties": {
                "name": int,
                "age": int,
                "score": float,
                "active": bool,
                "created_at": datetime,
            }
        },
        id="a-property-of-another-type",
    ),
]


class StoreSchemaCheckContract:
    @staticmethod
    async def build(
        store: VectorStore, vector_store_name: str, **schema_changes: Any
    ) -> VectorStore:
        raise NotImplementedError

    @pytest.mark.asyncio
    async def test_startup_refuses_a_store_migrate_has_not_created(self, store):
        unmigrated = await self.build(store, "unmigrated")

        with pytest.raises(SchemaMismatchError, match="memmachine-server migrate"):
            await unmigrated.startup()

    @pytest.mark.asyncio
    async def test_a_store_of_the_migrated_schema_starts(self, store):
        same = await self.build(store, store._vector_store_name)

        await same.startup()
        await same.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("schema_changes", _OTHER_SCHEMAS)
    async def test_startup_refuses_a_store_of_another_schema(
        self, store, schema_changes
    ):
        other = await self.build(store, store._vector_store_name, **schema_changes)

        with pytest.raises(SchemaMismatchError):
            await other.startup()

    @pytest.mark.asyncio
    async def test_migrate_refuses_a_store_of_other_dimensions(self, store):
        other = await self.build(store, store._vector_store_name, vector_dimensions=4)

        with pytest.raises(SchemaMismatchError):
            await other.migrate()

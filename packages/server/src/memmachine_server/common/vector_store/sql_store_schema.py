"""
A SQLite-backed store's schema, recorded by its migration.

The SQLite-backed stores keep vectors and properties per partition, so
nothing they share says what dimensions, metric, and declared properties a
store holds. Two tables in the store's catalog record them: one row per
store, and one row per property it declares.
"""

from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    insert,
    select,
)
from sqlalchemy.ext.asyncio import AsyncConnection

from memmachine_server.common.data_types import PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME

from .data_types import _IDENTIFIER_MAX_BYTES, VectorStoreSchema

# Long enough for every similarity metric's value and property type's name.
_NAME_MAX_LENGTH = 16


def store_schema_tables(metadata: MetaData, table_prefix: str) -> tuple[Table, Table]:
    """The table of stores and the table of their declared properties, in a catalog's metadata."""
    store_table = Table(
        f"{table_prefix}_st",
        metadata,
        Column("vector_store_name", String(_IDENTIFIER_MAX_BYTES), primary_key=True),
        Column("vector_dimensions", Integer, nullable=False),
        Column("similarity_metric", String(_NAME_MAX_LENGTH), nullable=False),
    )
    declared_property_table = Table(
        f"{table_prefix}_dp",
        metadata,
        Column(
            "vector_store_name",
            String(_IDENTIFIER_MAX_BYTES),
            ForeignKey(store_table.c.vector_store_name, ondelete="CASCADE"),
            primary_key=True,
        ),
        Column("property_key", String(_IDENTIFIER_MAX_BYTES), primary_key=True),
        Column("property_type", String(_NAME_MAX_LENGTH), nullable=False),
    )
    return store_table, declared_property_table


async def record_store_schema(
    connection: AsyncConnection,
    tables: tuple[Table, Table],
    vector_store_name: str,
    store_schema: VectorStoreSchema,
) -> None:
    """Record a store's schema in `tables`, unless a schema is recorded for it already."""
    store_table, declared_property_table = tables
    recorded = await connection.scalar(
        select(store_table.c.vector_store_name).where(
            store_table.c.vector_store_name == vector_store_name
        )
    )
    if recorded is not None:
        return
    await connection.execute(
        insert(store_table).values(
            vector_store_name=vector_store_name,
            vector_dimensions=store_schema.vector_dimensions,
            similarity_metric=store_schema.similarity_metric.value,
        )
    )
    if store_schema.indexed_properties:
        await connection.execute(
            insert(declared_property_table),
            [
                {
                    "vector_store_name": vector_store_name,
                    "property_key": key,
                    "property_type": PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[property_type],
                }
                for key, property_type in store_schema.indexed_properties.items()
            ],
        )

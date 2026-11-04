"""
A SQLite-backed store's schema, recorded by its migration and checked at startup.

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
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from memmachine_server.common.data_types import PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME
from memmachine_server.common.errors import SchemaMismatchError

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


async def check_store_schema(
    engine: AsyncEngine,
    tables: tuple[Table, Table],
    vector_store_name: str,
    store_schema: VectorStoreSchema,
) -> None:
    """
    Check that the schema recorded in `tables` for a store is the store's own.

    Raises:
        SchemaMismatchError:
            If no schema is recorded for the store, or the recorded one has
            other dimensions, another metric, or other declared properties.
    """
    store_table, declared_property_table = tables
    async with engine.connect() as connection:
        recorded = (
            await connection.execute(
                select(
                    store_table.c.vector_dimensions, store_table.c.similarity_metric
                ).where(store_table.c.vector_store_name == vector_store_name)
            )
        ).one_or_none()
        recorded_properties = dict(
            (
                await connection.execute(
                    select(
                        declared_property_table.c.property_key,
                        declared_property_table.c.property_type,
                    ).where(
                        declared_property_table.c.vector_store_name == vector_store_name
                    )
                )
            )
            .tuples()
            .all()
        )

    location = engine.url.render_as_string(hide_password=True)
    if recorded is None:
        raise SchemaMismatchError(
            f"The schema of the vector store {vector_store_name!r} in {location} "
            "is missing. Run `memmachine-server migrate`."
        )
    if tuple(recorded) != (
        store_schema.vector_dimensions,
        store_schema.similarity_metric.value,
    ):
        raise SchemaMismatchError(
            f"The vector store {vector_store_name!r} in {location} holds vectors "
            f"other than this store's: {store_schema.vector_dimensions} dimensions "
            f"by {store_schema.similarity_metric.value}."
        )
    declared_properties = {
        key: PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[property_type]
        for key, property_type in store_schema.indexed_properties.items()
    }
    if recorded_properties != declared_properties:
        raise SchemaMismatchError(
            f"The vector store {vector_store_name!r} in {location} declares other "
            "properties than this store."
        )

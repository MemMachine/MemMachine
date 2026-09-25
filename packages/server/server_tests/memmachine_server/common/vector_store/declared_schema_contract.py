"""
The declared-schema contract every vector store collection satisfies.

Each store's test module mixes `DeclaredSchemaContract` into a test class
and supplies the `collection` fixture: a collection of a store declaring
`INDEXED_PROPERTIES` (`name: str`, `age: int`, `score: float`,
`active: bool`, `created_at: datetime`). Ranking is asserted as a contract
(which records a filtered search admits), never as an exact list.

What a filter admits is only observable through a search, and an
approximate index can miss a record the filter admits. The contract is
therefore stated over fixtures every backend searches exactly: a handful
of records, far fewer than the query limit, which a brute-force store scans
whole, a graph index visits whole, and Qdrant and Milvus keep below their
indexing thresholds. `_store` checks that precondition after every upsert,
so a backend that indexes approximately from the first record fails on
recall, by name, and not on the filter.
"""

import math
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    In,
    IsNull,
    Not,
    Or,
)
from memmachine_server.common.vector_store import (
    PropertyTypeMismatchError,
    Record,
    UndeclaredPropertyKeyError,
)

_LIMIT = 100

_INSTANT = datetime(2024, 5, 1, 12, 0, 0, tzinfo=UTC)

_DECLARED = {
    "name": "alice",
    "age": 30,
    "score": 1.5,
    "active": True,
    "created_at": _INSTANT,
}


def _unit(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(x * x for x in vector))
    return [x / magnitude for x in vector]


def _record(vector: list[float], **properties) -> Record:
    return Record(uuid=uuid4(), vector=_unit(vector), properties=properties)


def _presence_records() -> list[Record]:
    """One record holding every declared key, one lacking each key, and one bare."""
    records = [_record([1.0, 0.0, 0.0], **_DECLARED)]
    for index, absent in enumerate(_DECLARED, start=1):
        held = {key: value for key, value in _DECLARED.items() if key != absent}
        records.append(_record([1.0, 0.1 * index, 0.0], **held))
    records.append(_record([0.0, 1.0, 0.0]))
    return records


async def _store(collection, records: list[Record]) -> None:
    """Upsert the fixture and check the collection searches it exactly.

    The filtered assertions below mean nothing if an unfiltered search
    already misses a record, so that is checked first and named when it
    fails.
    """
    await collection.upsert(records=records)
    [everything] = await collection.query(
        query_vectors=[_unit([1.0, 0.0, 0.0])], limit=_LIMIT
    )
    assert {match.record_uuid for match in everything.matches} == {
        record.uuid for record in records
    }, "an unfiltered search must reach every fixture record (exact recall)"


async def _admitted(collection, property_filter, *, limit=_LIMIT) -> set:
    [result] = await collection.query(
        query_vectors=[_unit([1.0, 0.0, 0.0])],
        limit=limit,
        property_filter=property_filter,
    )
    return {match.record_uuid for match in result.matches}


class DeclaredSchemaContract:
    """Mix into a store's test class; requires an async `collection` fixture."""

    @pytest.mark.asyncio
    async def test_upsert_rejects_an_undeclared_key(self, collection):
        with pytest.raises(UndeclaredPropertyKeyError, match="color"):
            await collection.upsert(records=[_record([1.0, 0.0, 0.0], color="red")])

    @pytest.mark.asyncio
    async def test_upsert_rejects_a_value_of_another_type(self, collection):
        with pytest.raises(PropertyTypeMismatchError, match="age"):
            await collection.upsert(records=[_record([1.0, 0.0, 0.0], age="5")])
        # A bool is an int at runtime and is still not one here.
        with pytest.raises(PropertyTypeMismatchError, match="age"):
            await collection.upsert(records=[_record([1.0, 0.0, 0.0], age=True)])

    @pytest.mark.asyncio
    async def test_query_rejects_an_undeclared_key(self, collection):
        with pytest.raises(UndeclaredPropertyKeyError, match="color"):
            await _admitted(collection, Comparison(field="color", op="=", value="red"))
        with pytest.raises(UndeclaredPropertyKeyError, match="color"):
            await _admitted(
                collection,
                And(
                    left=Comparison(field="name", op="=", value="a"),
                    right=IsNull(field="color"),
                ),
            )

    @pytest.mark.asyncio
    async def test_every_supported_node_evaluates_during_a_search(self, collection):
        held = _record([1.0, 0.0, 0.0], name="alice", age=30, score=1.5, active=True)
        other = _record([0.0, 1.0, 0.0], name="bob", age=40, active=False)
        lacking = _record([0.0, 0.0, 1.0], active=False)
        await _store(collection, [held, other, lacking])

        expected = {
            Comparison: (Comparison(field="age", op=">", value=35), {other.uuid}),
            In: (In(field="name", values=["alice", "carol"]), {held.uuid}),
            IsNull: (IsNull(field="name"), {lacking.uuid}),
            And: (
                And(
                    left=Comparison(field="name", op="=", value="alice"),
                    right=Comparison(field="age", op="=", value=30),
                ),
                {held.uuid},
            ),
            Or: (
                Or(
                    left=Comparison(field="name", op="=", value="alice"),
                    right=Comparison(field="name", op="=", value="bob"),
                ),
                {held.uuid, other.uuid},
            ),
            # Every record holds `active`, so the negation's answer does not
            # depend on how a backend treats a record lacking the field.
            Not: (
                Not(Comparison(field="active", op="=", value=True)),
                {other.uuid, lacking.uuid},
            ),
        }
        assert set(expected) >= collection.supported_filter_nodes
        for node, (property_filter, admitted) in expected.items():
            assert node in collection.supported_filter_nodes
            assert await _admitted(collection, property_filter) == admitted, node

    @pytest.mark.asyncio
    async def test_is_null_matches_the_records_without_the_key(self, collection):
        holding = _record([1.0, 0.0, 0.0], name="alice", age=30)
        lacking = _record([0.0, 1.0, 0.0], age=31)
        await _store(collection, [holding, lacking])

        assert await _admitted(collection, IsNull(field="name")) == {lacking.uuid}
        assert await _admitted(collection, Not(IsNull(field="name"))) == {holding.uuid}

    @pytest.mark.asyncio
    async def test_datetime_bounds_hold_at_microsecond_precision(self, collection):
        base = datetime(2024, 3, 1, 12, 0, 0, tzinfo=UTC)
        instants = [base + timedelta(microseconds=offset) for offset in range(3)]
        records = [
            _record([1.0, 0.0, 0.0], created_at=instants[0]),
            _record([0.0, 1.0, 0.0], created_at=instants[1]),
            _record([0.0, 0.0, 1.0], created_at=instants[2]),
        ]
        await _store(collection, records)

        assert await _admitted(
            collection, Comparison(field="created_at", op=">=", value=instants[1])
        ) == {records[1].uuid, records[2].uuid}
        assert await _admitted(
            collection, Comparison(field="created_at", op="<", value=instants[1])
        ) == {records[0].uuid}
        assert await _admitted(
            collection, Comparison(field="created_at", op="=", value=instants[1])
        ) == {records[1].uuid}

    @pytest.mark.asyncio
    async def test_a_bound_in_another_zone_is_the_same_instant(self, collection):
        instant = datetime(2024, 3, 1, 12, 0, 0, tzinfo=UTC)
        record = _record([1.0, 0.0, 0.0], created_at=instant)
        await _store(collection, [record])

        from datetime import timezone

        same_instant = instant.astimezone(timezone(timedelta(hours=-8)))
        assert await _admitted(
            collection, Comparison(field="created_at", op="=", value=same_instant)
        ) == {record.uuid}

    @pytest.mark.asyncio
    async def test_the_filter_is_evaluated_during_the_search(self, collection):
        """A filtered search finds an admitted record behind nearer excluded ones."""
        near = [_record([1.0, 0.01 * i, 0.0], name="near") for i in range(1, 6)]
        far = _record([0.0, 1.0, 0.0], name="far")
        await _store(collection, [*near, far])

        assert await _admitted(
            collection, Comparison(field="name", op="=", value="far"), limit=1
        ) == {far.uuid}

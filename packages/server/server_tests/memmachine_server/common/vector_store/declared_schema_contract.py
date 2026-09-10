"""
The declared-schema contract every vector store collection satisfies.

Each store's test module mixes `DeclaredSchemaContract` into a test class
and supplies the `collection` fixture: a collection of a store declaring
`INDEXED_PROPERTIES` (`name: str`, `age: int`, `score: float`,
`active: bool`, `created_at: datetime`). Ranking is asserted as a contract
(which records a filtered search admits), never as an exact list.
"""

import math
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from memmachine_server.common.filter import (
    And,
    Equals,
    In,
    IsMissing,
    Not,
    NotEquals,
    Or,
    Ordering,
)
from memmachine_server.common.vector_store import (
    PropertyTypeMismatchError,
    Record,
    UndeclaredPropertyKeyError,
)

_LIMIT = 100


def _unit(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(x * x for x in vector))
    return [x / magnitude for x in vector]


def _record(vector: list[float], **properties) -> Record:
    return Record(uuid=uuid4(), vector=_unit(vector), properties=properties)


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
            await _admitted(collection, Equals(field="color", value="red"))
        with pytest.raises(UndeclaredPropertyKeyError, match="color"):
            await _admitted(
                collection,
                And((Equals(field="name", value="a"), IsMissing(field="color"))),
            )

    @pytest.mark.asyncio
    async def test_every_supported_node_evaluates_during_a_search(self, collection):
        held = _record([1.0, 0.0, 0.0], name="alice", age=30, score=1.5, active=True)
        other = _record([0.0, 1.0, 0.0], name="bob", age=40)
        lacking = _record([0.0, 0.0, 1.0])
        await collection.upsert(records=[held, other, lacking])

        expected = {
            Equals: (Equals(field="name", value="alice"), {held.uuid}),
            NotEquals: (NotEquals(field="name", value="alice"), {other.uuid}),
            Ordering: (Ordering(field="age", op=">", value=35), {other.uuid}),
            In: (In(field="name", values=("alice", "carol")), {held.uuid}),
            IsMissing: (IsMissing(field="name"), {lacking.uuid}),
            And: (
                And(
                    (Equals(field="name", value="alice"), Equals(field="age", value=30))
                ),
                {held.uuid},
            ),
            Or: (
                Or(
                    (
                        Equals(field="name", value="alice"),
                        Equals(field="name", value="bob"),
                    )
                ),
                {held.uuid, other.uuid},
            ),
            Not: (Not(Equals(field="name", value="bob")), {held.uuid, lacking.uuid}),
        }
        assert set(expected) >= collection.supported_filter_nodes
        for node, (property_filter, admitted) in expected.items():
            assert node in collection.supported_filter_nodes
            assert await _admitted(collection, property_filter) == admitted, node

    @pytest.mark.asyncio
    async def test_not_equals_excludes_absence_and_not_equals_includes_it(
        self, collection
    ):
        holding = _record([1.0, 0.0, 0.0], name="alice")
        differing = _record([0.0, 1.0, 0.0], name="bob")
        lacking = _record([0.0, 0.0, 1.0])
        await collection.upsert(records=[holding, differing, lacking])

        assert await _admitted(collection, NotEquals(field="name", value="alice")) == {
            differing.uuid
        }
        assert await _admitted(
            collection, Not(Equals(field="name", value="alice"))
        ) == {
            differing.uuid,
            lacking.uuid,
        }

    @pytest.mark.asyncio
    async def test_is_missing_matches_absence(self, collection):
        holding = _record([1.0, 0.0, 0.0], name="alice", age=30)
        lacking = _record([0.0, 1.0, 0.0], age=31)
        await collection.upsert(records=[holding, lacking])

        assert await _admitted(collection, IsMissing(field="name")) == {lacking.uuid}
        assert await _admitted(collection, Not(IsMissing(field="name"))) == {
            holding.uuid
        }

    @pytest.mark.asyncio
    async def test_a_predicate_of_another_type_matches_nothing(self, collection):
        held = _record([1.0, 0.0, 0.0], name="5", age=5)
        await collection.upsert(records=[held])

        assert await _admitted(collection, Equals(field="age", value="5")) == set()
        assert await _admitted(collection, Equals(field="name", value=5)) == set()
        assert await _admitted(collection, Equals(field="age", value=5)) == {held.uuid}

    @pytest.mark.asyncio
    async def test_datetime_bounds_hold_at_microsecond_precision(self, collection):
        base = datetime(2024, 3, 1, 12, 0, 0, tzinfo=UTC)
        instants = [base + timedelta(microseconds=offset) for offset in range(3)]
        records = [
            _record([1.0, 0.0, 0.0], created_at=instants[0]),
            _record([0.0, 1.0, 0.0], created_at=instants[1]),
            _record([0.0, 0.0, 1.0], created_at=instants[2]),
        ]
        await collection.upsert(records=records)

        assert await _admitted(
            collection, Ordering(field="created_at", op=">=", value=instants[1])
        ) == {records[1].uuid, records[2].uuid}
        assert await _admitted(
            collection, Ordering(field="created_at", op="<", value=instants[1])
        ) == {records[0].uuid}
        assert await _admitted(
            collection, Equals(field="created_at", value=instants[1])
        ) == {records[1].uuid}

    @pytest.mark.asyncio
    async def test_a_bound_in_another_zone_is_the_same_instant(self, collection):
        instant = datetime(2024, 3, 1, 12, 0, 0, tzinfo=UTC)
        record = _record([1.0, 0.0, 0.0], created_at=instant)
        await collection.upsert(records=[record])

        from datetime import timezone

        same_instant = instant.astimezone(timezone(timedelta(hours=-8)))
        assert await _admitted(
            collection, Equals(field="created_at", value=same_instant)
        ) == {record.uuid}

    @pytest.mark.asyncio
    async def test_the_filter_is_evaluated_during_the_search(self, collection):
        """A filtered search finds an admitted record behind nearer excluded ones."""
        near = [_record([1.0, 0.01 * i, 0.0], name="near") for i in range(1, 6)]
        far = _record([0.0, 1.0, 0.0], name="far")
        await collection.upsert(records=[*near, far])

        assert await _admitted(
            collection, Equals(field="name", value="far"), limit=1
        ) == {far.uuid}

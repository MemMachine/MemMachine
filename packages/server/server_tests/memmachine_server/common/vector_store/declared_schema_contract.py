"""
The declared-schema contract every vector store partition satisfies.

Each store's test module mixes `DeclaredSchemaContract` into a test class
and supplies the `partition` fixture: a partition of a store declaring
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
from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest

from memmachine_server.common.filter import (
    And,
    Equals,
    In,
    IsNull,
    Not,
    Or,
    Ordering,
    Property,
    SystemField,
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

_COMPLEMENT_FILTERS = [
    Equals(field=Property("name"), value="alice"),
    Equals(field=Property("active"), value=True),
    Equals(field=Property("created_at"), value=_INSTANT),
    Ordering(field=Property("age"), op=">", value=20),
    Ordering(field=Property("score"), op="<=", value=1.5),
    Ordering(field=Property("created_at"), op=">=", value=_INSTANT),
    In(field=Property("name"), values=("alice", "carol")),
    In(field=Property("name"), values=()),
    IsNull(field=Property("name")),
    IsNull(field=Property("score")),
    And(
        (
            Equals(field=Property("name"), value="alice"),
            Ordering(field=Property("age"), op=">", value=20),
        )
    ),
    Or((Equals(field=Property("name"), value="alice"), IsNull(field=Property("name")))),
    Not(Equals(field=Property("name"), value="alice")),
]


_MISTYPED_FILTERS = [
    Equals(field=Property("age"), value="30"),
    Equals(field=Property("name"), value=30),
    Equals(field=Property("active"), value=1),
    Equals(field=Property("score"), value=True),
    Equals(field=Property("created_at"), value="2024-05-01T12:00:00+00:00"),
    Equals(field=Property("age"), value=30.0),
    Ordering(field=Property("age"), op="<", value=30.5),
    Ordering(field=Property("created_at"), op="<", value=1),
    Ordering(field=Property("name"), op="<", value=1),
    In(field=Property("age"), values=("30",)),
    In(field=Property("name"), values=(30,)),
    In(field=Property("active"), values=(1,)),
    Not(
        And(
            (
                Equals(field=Property("name"), value="alice"),
                Equals(field=Property("age"), value="30"),
            )
        )
    ),
    Equals(field=Property("age"), value=2**63),
    Ordering(field=Property("age"), op="<", value=-(2**63) - 1),
    In(field=Property("age"), values=(1, 2**63)),
    Ordering(field=Property("score"), op=">", value=2**63),
]

_MISTYPED_PROPERTIES = [
    ("age", "30"),
    # A bool is an int at runtime and is still not one here.
    ("age", True),
    ("age", 30.5),
    ("score", True),
    ("score", "1.5"),
    ("score", math.nan),
    ("score", -math.inf),
    ("age", 2**63),
    ("age", -(2**63) - 1),
    ("score", 2**63),
    ("name", 5),
    ("active", 1),
    ("created_at", "2024-05-01T12:00:00+00:00"),
]

_NON_FINITE_FILTERS = [
    Equals(field=Property("score"), value=math.nan),
    Equals(field=Property("score"), value=-math.inf),
    Ordering(field=Property("score"), op="<", value=math.inf),
    Or(
        (
            IsNull(field=Property("name")),
            Ordering(field=Property("score"), op=">", value=math.nan),
        )
    ),
]


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


async def _admitted(partition, property_filter, *, limit=_LIMIT) -> set:
    [result] = await partition.query(
        query_vectors=[_unit([1.0, 0.0, 0.0])],
        limit=limit,
        property_filter=property_filter,
    )
    return {match.record_uuid for match in result.matches}


class DeclaredSchemaContract:
    """Mixed into a store's test class, which supplies an async `partition` fixture and the `settle` and `stored_value` hooks."""

    @staticmethod
    async def settle(partition) -> None:
        """Return once the store's reads reflect every write made so far."""
        raise NotImplementedError

    @staticmethod
    async def stored_value(partition, record_uuid, key):
        """The value the backend holds for a record's property, read past the store."""
        raise NotImplementedError

    async def _store(self, partition, records: list[Record]) -> None:
        """Upsert the fixture and check the partition searches it exactly.

        The filtered assertions below mean nothing if an unfiltered search
        already misses a record, so that is checked first and named when it
        fails.
        """
        await partition.upsert(records=records)
        await self.settle(partition)
        [everything] = await partition.query(
            query_vectors=[_unit([1.0, 0.0, 0.0])], limit=_LIMIT
        )
        assert {match.record_uuid for match in everything.matches} == {
            record.uuid for record in records
        }, "an unfiltered search must reach every fixture record (exact recall)"

    @pytest.mark.asyncio
    async def test_upsert_rejects_an_undeclared_key(self, partition):
        with pytest.raises(UndeclaredPropertyKeyError, match="color"):
            await partition.upsert(records=[_record([1.0, 0.0, 0.0], color="red")])

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("key", "value"), _MISTYPED_PROPERTIES)
    async def test_upsert_rejects_a_value_of_another_type(self, partition, key, value):
        with pytest.raises(PropertyTypeMismatchError, match=key):
            await partition.upsert(records=[_record([1.0, 0.0, 0.0], **{key: value})])

    @pytest.mark.asyncio
    async def test_upsert_rejects_a_batch_naming_a_record_uuid_twice(self, partition):
        first = _record([1.0, 0.0, 0.0], name="first")
        other = _record([0.0, 1.0, 0.0], name="other")
        again = Record(
            uuid=first.uuid, vector=_unit([0.0, 0.0, 1.0]), properties={"name": "again"}
        )
        with pytest.raises(ValueError, match=f"{first.uuid} occurs more than once"):
            await partition.upsert(records=[first, other, again])

        await self.settle(partition)
        [result] = await partition.query(
            query_vectors=[_unit([1.0, 0.0, 0.0])], limit=_LIMIT
        )
        assert result.matches == []

    @pytest.mark.asyncio
    async def test_an_int_written_to_a_float_key_is_stored_as_the_float_it_equals(
        self, partition
    ):
        two, three = (
            _record([1.0, 0.0, 0.0], score=2),
            _record([1.0, 0.1, 0.0], score=3),
        )
        await self._store(partition, [two, three])

        stored = await self.stored_value(partition, two.uuid, "score")
        assert type(stored) is float
        assert stored == 2.0
        assert await _admitted(
            partition, Equals(field=Property("score"), value=2.0)
        ) == {two.uuid}
        assert await _admitted(
            partition, Ordering(field=Property("score"), op=">", value=2.5)
        ) == {three.uuid}

    @pytest.mark.asyncio
    async def test_query_rejects_an_undeclared_key(self, partition):
        with pytest.raises(UndeclaredPropertyKeyError, match="color"):
            await _admitted(partition, Equals(field=Property("color"), value="red"))
        with pytest.raises(UndeclaredPropertyKeyError, match="color"):
            await _admitted(
                partition,
                And(
                    (
                        Equals(field=Property("name"), value="a"),
                        IsNull(field=Property("color")),
                    )
                ),
            )

    @pytest.mark.asyncio
    async def test_query_rejects_a_system_field(self, partition):
        with pytest.raises(ValueError, match="system field 'name'"):
            await _admitted(partition, Equals(SystemField("name"), "alice"))
        with pytest.raises(ValueError, match="system field 'name'"):
            await _admitted(partition, Not(IsNull(SystemField("name"))))

    @pytest.mark.asyncio
    async def test_every_node_admits_the_matching_records(self, partition):
        held = _record([1.0, 0.0, 0.0], name="alice", age=30, score=1.5, active=True)
        other = _record([0.0, 1.0, 0.0], name="bob", age=40)
        lacking = _record([0.0, 0.0, 1.0])
        await self._store(partition, [held, other, lacking])

        expected = {
            Equals: (Equals(field=Property("name"), value="alice"), {held.uuid}),
            Ordering: (Ordering(field=Property("age"), op=">", value=35), {other.uuid}),
            In: (In(field=Property("name"), values=("alice", "carol")), {held.uuid}),
            IsNull: (IsNull(field=Property("name")), {lacking.uuid}),
            And: (
                And(
                    (
                        Equals(field=Property("name"), value="alice"),
                        Equals(field=Property("age"), value=30),
                    )
                ),
                {held.uuid},
            ),
            Or: (
                Or(
                    (
                        Equals(field=Property("name"), value="alice"),
                        Equals(field=Property("name"), value="bob"),
                    )
                ),
                {held.uuid, other.uuid},
            ),
            Not: (
                Not(Equals(field=Property("name"), value="bob")),
                {held.uuid, lacking.uuid},
            ),
        }
        for node, (property_filter, admitted) in expected.items():
            assert await _admitted(partition, property_filter) == admitted, node

    @pytest.mark.asyncio
    @pytest.mark.parametrize("property_filter", _COMPLEMENT_FILTERS)
    async def test_negation_is_the_complement(self, partition, property_filter):
        """`Not` admits every record the operand does not, absence included."""
        records = _presence_records()
        await self._store(partition, records)
        everything = {record.uuid for record in records}

        admitted = await _admitted(partition, property_filter)
        assert await _admitted(partition, Not(property_filter)) == everything - admitted
        assert await _admitted(partition, Not(Not(property_filter))) == admitted

    @pytest.mark.asyncio
    async def test_in_over_no_values_admits_nothing(self, partition):
        """An empty `In` is false, so its complement is every record."""
        records = _presence_records()
        await self._store(partition, records)

        assert (
            await _admitted(partition, In(field=Property("name"), values=())) == set()
        )
        assert await _admitted(
            partition, Not(In(field=Property("name"), values=()))
        ) == {record.uuid for record in records}

    @pytest.mark.asyncio
    async def test_is_null_matches_the_records_without_the_key(self, partition):
        holding = _record([1.0, 0.0, 0.0], name="alice", age=30)
        lacking = _record([0.0, 1.0, 0.0], age=31)
        await self._store(partition, [holding, lacking])

        assert await _admitted(partition, IsNull(field=Property("name"))) == {
            lacking.uuid
        }
        assert await _admitted(partition, Not(IsNull(field=Property("name")))) == {
            holding.uuid
        }

    @pytest.mark.asyncio
    async def test_a_reupsert_clears_the_keys_it_drops(self, partition):
        record = _record([1.0, 0.0, 0.0], **_DECLARED)
        await self._store(partition, [record])
        await partition.upsert(
            records=[
                Record(
                    uuid=record.uuid, vector=record.vector, properties={"name": "alice"}
                )
            ]
        )
        await self.settle(partition)
        for key in ("age", "score", "active", "created_at"):
            assert await _admitted(partition, IsNull(field=Property(key))) == {
                record.uuid
            }, key

    @pytest.mark.asyncio
    async def test_negated_equality_keeps_a_record_without_the_key(self, partition):
        holding = _record([1.0, 0.0, 0.0], name="alice")
        differing = _record([0.0, 1.0, 0.0], name="bob")
        lacking = _record([0.0, 0.0, 1.0])
        await self._store(partition, [holding, differing, lacking])

        assert await _admitted(
            partition, Not(Equals(field=Property("name"), value="alice"))
        ) == {
            differing.uuid,
            lacking.uuid,
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("property_filter", _MISTYPED_FILTERS)
    async def test_query_rejects_a_value_of_another_type(
        self, partition, property_filter
    ):
        await self._store(partition, [_record([1.0, 0.0, 0.0], **_DECLARED)])

        with pytest.raises(PropertyTypeMismatchError):
            await _admitted(partition, property_filter)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("property_filter", _NON_FINITE_FILTERS)
    async def test_query_rejects_a_float_that_is_not_finite(
        self, partition, property_filter
    ):
        await self._store(partition, [_record([1.0, 0.0, 0.0], **_DECLARED)])

        with pytest.raises(PropertyTypeMismatchError, match="score"):
            await _admitted(partition, property_filter)

    @pytest.mark.asyncio
    async def test_an_int_compares_with_a_float_key_as_the_float_it_equals(
        self, partition
    ):
        above, equal, below = (
            _record([1.0, 0.0, 0.0], score=3.5),
            _record([1.0, 0.1, 0.0], score=3.0),
            _record([1.0, 0.2, 0.0], score=2.5),
        )
        bare = _record([0.0, 1.0, 0.0])
        await self._store(partition, [above, equal, below, bare])
        everything = {above.uuid, equal.uuid, below.uuid, bare.uuid}

        expected = [
            (Ordering(field=Property("score"), op=">", value=3), {above.uuid}),
            (
                Ordering(field=Property("score"), op=">=", value=3),
                {above.uuid, equal.uuid},
            ),
            (Ordering(field=Property("score"), op="<", value=3), {below.uuid}),
            (Equals(field=Property("score"), value=3), {equal.uuid}),
        ]
        for property_filter, admitted in expected:
            assert await _admitted(partition, property_filter) == admitted, (
                property_filter
            )
            assert (
                await _admitted(partition, Not(property_filter))
                == everything - admitted
            ), property_filter

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "property_filter",
        [
            In(field=Property("score"), values=(1, 2)),
            In(field=Property("score"), values=()),
            Not(In(field=Property("created_at"), values=(1,))),
        ],
    )
    async def test_query_rejects_a_membership_test_on_a_float_or_datetime_key(
        self, partition, property_filter
    ):
        with pytest.raises(ValueError, match="membership test"):
            await _admitted(partition, property_filter)

    @pytest.mark.asyncio
    async def test_a_float_equals_only_itself(self, partition):
        scores = [0.1, 0.30000000000000004, 0.3, 1e16, 1e16 + 2.0, -2.5]
        records = [
            _record([1.0, 0.01 * index, 0.0], score=score)
            for index, score in enumerate(scores)
        ]
        await self._store(partition, records)

        for record in records:
            score = record.properties["score"]
            assert await _admitted(
                partition, Equals(field=Property("score"), value=score)
            ) == {record.uuid}, score

    @pytest.mark.asyncio
    async def test_an_int_compares_across_64_signed_bits(self, partition):
        lowest = _record([1.0, 0.0, 0.0], age=-(2**63))
        highest = _record([1.0, 0.1, 0.0], age=2**63 - 1)
        await self._store(partition, [lowest, highest])

        expected = [
            (Equals(field=Property("age"), value=2**63 - 1), {highest.uuid}),
            (Equals(field=Property("age"), value=-(2**63)), {lowest.uuid}),
            (Ordering(field=Property("age"), op=">=", value=2**63 - 1), {highest.uuid}),
            (Ordering(field=Property("age"), op="<=", value=-(2**63)), {lowest.uuid}),
            (Ordering(field=Property("age"), op=">", value=-(2**63)), {highest.uuid}),
            (
                In(field=Property("age"), values=(-(2**63), 2**63 - 1)),
                {lowest.uuid, highest.uuid},
            ),
        ]
        for property_filter, admitted in expected:
            assert await _admitted(partition, property_filter) == admitted, (
                property_filter
            )

    @pytest.mark.asyncio
    async def test_a_string_matches_whatever_its_characters(self, partition):
        names = ["café", "smile 😀", 'say "hi" \\ back', "line\nbreak\ttab", ""]
        records = [
            _record([1.0, 0.1 * index, 0.0], name=name)
            for index, name in enumerate(names)
        ]
        await self._store(partition, records)

        for record in records:
            assert await _admitted(
                partition,
                Equals(field=Property("name"), value=record.properties["name"]),
            ) == {record.uuid}, record.properties["name"]
        assert await _admitted(
            partition,
            In(field=Property("name"), values=("smile 😀", 'say "hi" \\ back')),
        ) == {records[1].uuid, records[2].uuid}

    @pytest.mark.asyncio
    async def test_datetime_bounds_hold_at_microsecond_precision(self, partition):
        base = datetime(2024, 3, 1, 12, 0, 0, tzinfo=UTC)
        instants = [base + timedelta(microseconds=offset) for offset in range(3)]
        records = [
            _record([1.0, 0.0, 0.0], created_at=instants[0]),
            _record([0.0, 1.0, 0.0], created_at=instants[1]),
            _record([0.0, 0.0, 1.0], created_at=instants[2]),
        ]
        await self._store(partition, records)

        assert await _admitted(
            partition,
            Ordering(field=Property("created_at"), op=">=", value=instants[1]),
        ) == {records[1].uuid, records[2].uuid}
        assert await _admitted(
            partition, Ordering(field=Property("created_at"), op="<", value=instants[1])
        ) == {records[0].uuid}
        assert await _admitted(
            partition, Equals(field=Property("created_at"), value=instants[1])
        ) == {records[1].uuid}

    @pytest.mark.asyncio
    async def test_a_bound_in_another_zone_is_the_same_instant(self, partition):
        instant = datetime(2024, 3, 1, 12, 0, 0, tzinfo=UTC)
        record = _record([1.0, 0.0, 0.0], created_at=instant)
        await self._store(partition, [record])

        same_instant = instant.astimezone(timezone(timedelta(hours=-8)))
        assert await _admitted(
            partition, Equals(field=Property("created_at"), value=same_instant)
        ) == {record.uuid}

    @pytest.mark.asyncio
    async def test_the_filter_is_evaluated_during_the_search(self, partition):
        """A filtered search finds an admitted record behind nearer excluded ones."""
        near = [_record([1.0, 0.01 * i, 0.0], name="near") for i in range(1, 6)]
        far = _record([0.0, 1.0, 0.0], name="far")
        await self._store(partition, [*near, far])

        assert await _admitted(
            partition, Equals(field=Property("name"), value="far"), limit=1
        ) == {far.uuid}

"""Integration tests for compile_sql_filter with an in-memory SQLite database.

These tests verify that JSON metadata filtering handles numeric types correctly
(integer ordering, float ordering, boolean equality) rather than falling back
to string/lexicographic comparison.

Properties-JSON tests additionally verify the type-tagged {"t": …, "v": …} format
against both SQLite and PostgreSQL (via testcontainers).
"""

from datetime import UTC, datetime, timedelta, timezone
from typing import Any, cast

import pytest
import pytest_asyncio
from sqlalchemy import JSON, Column, Integer, String, create_engine, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Session

from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    In,
    IsNull,
    Not,
    Or,
    parse_filter,
)
from memmachine_server.common.filter.sql_filter_util import compile_sql_filter
from memmachine_server.common.properties_json import encode_properties

# ============================================================================
# Raw JSON ("json" kind) + direct column ("column" kind) tests — sync SQLite
# ============================================================================


class _JsonBase(DeclarativeBase):
    pass


class _Item(_JsonBase):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    json_metadata = Column(JSON, nullable=True)


def _resolve_json_field(field: str):
    """Field resolver for the Item model: column + json kinds."""
    normalized = field.lower()
    field_mapping = {
        "id": _Item.id.expression,
        "name": _Item.name.expression,
    }
    if normalized in field_mapping:
        return field_mapping[normalized], "column"

    if normalized.startswith(("m.", "metadata.")):
        key = normalized.split(".", 1)[1]
        return _Item.json_metadata[key], "json"

    raise ValueError(f"Unknown filter field: {field!r}")


@pytest.fixture
def json_session():
    """Create an in-memory SQLite database with seeded rows."""
    engine = create_engine("sqlite:///:memory:")
    _JsonBase.metadata.create_all(engine)
    with Session(engine) as s:
        s.add_all(
            [
                _Item(
                    name="alpha",
                    json_metadata={
                        "count": 5,
                        "score": 1.5,
                        "active": True,
                        "tag": "a",
                    },
                ),
                _Item(
                    name="beta",
                    json_metadata={
                        "count": 10,
                        "score": 2.5,
                        "active": False,
                        "tag": "b",
                    },
                ),
                _Item(
                    name="gamma",
                    json_metadata={
                        "count": 15,
                        "score": 3.5,
                        "active": True,
                        "tag": "c",
                    },
                ),
                _Item(
                    name="delta",
                    json_metadata={
                        "count": 20,
                        "score": 4.5,
                        "active": False,
                        "tag": "d",
                    },
                ),
                _Item(name="epsilon", json_metadata=None),
            ]
        )
        s.commit()
        yield s


def _query_json_names(session: Session, filter_str: str) -> set[str]:
    """Parse filter, compile to SQL, execute, and return set of matching names."""
    expr = parse_filter(filter_str)
    clause = compile_sql_filter(expr, _resolve_json_field)
    stmt = select(_Item.name).where(clause)
    return {row[0] for row in session.execute(stmt)}


# --- Integer ordering ---


def test_json_int_greater_than(json_session):
    assert _query_json_names(json_session, "m.count > 10") == {"gamma", "delta"}


def test_json_int_greater_equal(json_session):
    assert _query_json_names(json_session, "m.count >= 10") == {
        "beta",
        "gamma",
        "delta",
    }


def test_json_int_less_than(json_session):
    assert _query_json_names(json_session, "m.count < 10") == {"alpha"}


def test_json_int_less_equal(json_session):
    assert _query_json_names(json_session, "m.count <= 10") == {"alpha", "beta"}


# --- Integer equality ---


def test_json_int_equality(json_session):
    assert _query_json_names(json_session, "m.count = 10") == {"beta"}


def test_json_int_not_equal(json_session):
    assert _query_json_names(json_session, "m.count != 10") == {
        "alpha",
        "gamma",
        "delta",
    }


# --- Float ordering ---


def test_json_float_greater_than(json_session):
    assert _query_json_names(json_session, "m.score > 2.0") == {
        "beta",
        "gamma",
        "delta",
    }


def test_json_float_less_than(json_session):
    assert _query_json_names(json_session, "m.score < 3.0") == {"alpha", "beta"}


def test_json_float_less_equal(json_session):
    assert _query_json_names(json_session, "m.score <= 2.5") == {"alpha", "beta"}


# --- Boolean equality ---


def test_json_bool_true(json_session):
    assert _query_json_names(json_session, "m.active = true") == {"alpha", "gamma"}


def test_json_bool_false(json_session):
    assert _query_json_names(json_session, "m.active = false") == {"beta", "delta"}


# --- String equality ---


def test_json_string_equality(json_session):
    assert _query_json_names(json_session, "m.tag = 'a'") == {"alpha"}


def test_json_string_not_equal(json_session):
    assert _query_json_names(json_session, "m.tag != 'a'") == {
        "beta",
        "gamma",
        "delta",
    }


# --- IN ---


def test_json_int_in(json_session):
    assert _query_json_names(json_session, "m.count IN (5, 15)") == {"alpha", "gamma"}


def test_json_string_in(json_session):
    assert _query_json_names(json_session, "m.tag IN ('a', 'b')") == {"alpha", "beta"}


# --- IS NULL ---


def test_json_is_null(json_session):
    assert _query_json_names(json_session, "m.tag IS NULL") == {"epsilon"}


# --- NOT / NOT IN ---


def test_json_not_comparison(json_session):
    assert _query_json_names(json_session, "NOT m.count > 10") == {"alpha", "beta"}


def test_json_not_in(json_session):
    assert _query_json_names(json_session, "m.tag NOT IN ('a', 'b')") == {
        "gamma",
        "delta",
    }


# --- Non-metadata column ---


def test_column_equality(json_session):
    assert _query_json_names(json_session, "name = 'alpha'") == {"alpha"}


def test_column_in(json_session):
    assert _query_json_names(json_session, "name IN ('alpha', 'beta')") == {
        "alpha",
        "beta",
    }


# --- Compound ---


def test_json_not_and_compound(json_session):
    result = _query_json_names(json_session, "NOT (m.count > 10 AND m.active = true)")
    assert result == {"alpha", "beta", "delta"}


def test_json_or(json_session):
    assert _query_json_names(json_session, "m.tag = 'a' OR m.tag = 'c'") == {
        "alpha",
        "gamma",
    }


def test_json_not_or_compound(json_session):
    assert _query_json_names(json_session, "NOT (m.tag = 'a' OR m.tag = 'b')") == {
        "gamma",
        "delta",
    }


# --- Error paths ---


def test_unknown_field_raises(json_session):
    with pytest.raises(ValueError, match="Unknown filter field"):
        _query_json_names(json_session, "nonexistent = 1")


def test_unsupported_expr_type():
    with pytest.raises(TypeError, match="Unsupported filter expression type"):
        compile_sql_filter("bad", _resolve_json_field)


# ============================================================================
# Properties JSON ("properties_json" kind) tests — async, SQLite + PostgreSQL
# ============================================================================

_JSON_AUTO = JSON().with_variant(JSONB, "postgresql")


class _PropsBase(DeclarativeBase):
    pass


class _PropsRow(_PropsBase):
    __tablename__ = "test_properties_json_filters"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    properties = Column(_JSON_AUTO, nullable=False)


def _resolve_properties_json_field(field: str):
    return _PropsRow.properties[field], "properties_json"


@pytest_asyncio.fixture
async def properties_session(sqlalchemy_engine: AsyncEngine):
    """Create tables, seed properties-JSON rows, yield session, then drop tables."""
    async with sqlalchemy_engine.begin() as conn:
        await conn.run_sync(_PropsBase.metadata.create_all)

    est = timezone(timedelta(hours=-5))
    jst = timezone(timedelta(hours=9))

    async with AsyncSession(sqlalchemy_engine) as s:
        s.add_all(
            [
                _PropsRow(
                    name="alpha",
                    properties=encode_properties(
                        {
                            "count": 5,
                            "score": 1.5,
                            "active": True,
                            "tag": "a",
                            "ts": datetime(2024, 1, 10, 12, 0, 0, tzinfo=UTC),
                        }
                    ),
                ),
                _PropsRow(
                    name="beta",
                    properties=encode_properties(
                        {
                            "count": 10,
                            "score": 2.5,
                            "active": False,
                            "tag": "b",
                            "ts": datetime(2024, 3, 15, 8, 0, 0, tzinfo=est),
                        }
                    ),
                ),
                _PropsRow(
                    name="gamma",
                    properties=encode_properties(
                        {
                            "count": 15,
                            "score": 3.5,
                            "active": True,
                            "tag": "c",
                            "ts": datetime(2024, 6, 20, 21, 0, 0, tzinfo=jst),
                        }
                    ),
                ),
                _PropsRow(
                    name="delta",
                    properties=encode_properties(
                        {
                            "count": 20,
                            "score": 4.5,
                            "active": False,
                            "tag": "d",
                            "ts": datetime(2024, 9, 1, 0, 0, 0, tzinfo=UTC),
                        }
                    ),
                ),
                _PropsRow(
                    name="epsilon",
                    properties=encode_properties({"count": 0, "active": True}),
                ),
            ]
        )
        await s.commit()
        yield s

    async with sqlalchemy_engine.begin() as conn:
        await conn.run_sync(_PropsBase.metadata.drop_all)


async def _query_props_names(session: AsyncSession, expr) -> set[str]:
    clause = compile_sql_filter(expr, _resolve_properties_json_field)
    result = await session.execute(select(_PropsRow.name).where(clause))
    return {row[0] for row in result}


class TestPropsJsonIntFilters:
    @pytest.mark.asyncio
    async def test_eq(self, properties_session):
        assert await _query_props_names(
            properties_session, Comparison("count", "=", 10)
        ) == {"beta"}

    @pytest.mark.asyncio
    async def test_neq(self, properties_session):
        assert await _query_props_names(
            properties_session, Comparison("count", "!=", 10)
        ) == {
            "alpha",
            "gamma",
            "delta",
            "epsilon",
        }

    @pytest.mark.asyncio
    async def test_gt(self, properties_session):
        assert await _query_props_names(
            properties_session, Comparison("count", ">", 10)
        ) == {
            "gamma",
            "delta",
        }

    @pytest.mark.asyncio
    async def test_gte(self, properties_session):
        assert await _query_props_names(
            properties_session, Comparison("count", ">=", 10)
        ) == {
            "beta",
            "gamma",
            "delta",
        }

    @pytest.mark.asyncio
    async def test_lt(self, properties_session):
        assert await _query_props_names(
            properties_session, Comparison("count", "<", 10)
        ) == {
            "alpha",
            "epsilon",
        }

    @pytest.mark.asyncio
    async def test_lte(self, properties_session):
        assert await _query_props_names(
            properties_session, Comparison("count", "<=", 10)
        ) == {
            "alpha",
            "beta",
            "epsilon",
        }


class TestPropsJsonFloatFilters:
    @pytest.mark.asyncio
    async def test_gt(self, properties_session):
        assert await _query_props_names(
            properties_session, Comparison("score", ">", 2.0)
        ) == {
            "beta",
            "gamma",
            "delta",
        }

    @pytest.mark.asyncio
    async def test_lt(self, properties_session):
        assert await _query_props_names(
            properties_session, Comparison("score", "<", 3.0)
        ) == {
            "alpha",
            "beta",
        }


class TestPropsJsonBoolFilters:
    @pytest.mark.asyncio
    async def test_true(self, properties_session):
        assert await _query_props_names(
            properties_session, Comparison("active", "=", True)
        ) == {
            "alpha",
            "gamma",
            "epsilon",
        }

    @pytest.mark.asyncio
    async def test_false(self, properties_session):
        assert await _query_props_names(
            properties_session, Comparison("active", "=", False)
        ) == {
            "beta",
            "delta",
        }


class TestPropsJsonStringFilters:
    @pytest.mark.asyncio
    async def test_eq(self, properties_session):
        assert await _query_props_names(
            properties_session, Comparison("tag", "=", "a")
        ) == {"alpha"}

    @pytest.mark.asyncio
    async def test_neq(self, properties_session):
        assert await _query_props_names(
            properties_session, Comparison("tag", "!=", "a")
        ) == {
            "beta",
            "gamma",
            "delta",
        }


class TestPropsJsonDatetimeFilters:
    @pytest.mark.asyncio
    async def test_gt(self, properties_session):
        cutoff = datetime(2024, 6, 1, 0, 0, 0, tzinfo=UTC)
        assert await _query_props_names(
            properties_session, Comparison("ts", ">", cutoff)
        ) == {
            "gamma",
            "delta",
        }

    @pytest.mark.asyncio
    async def test_lt(self, properties_session):
        cutoff = datetime(2024, 6, 1, 0, 0, 0, tzinfo=UTC)
        assert await _query_props_names(
            properties_session, Comparison("ts", "<", cutoff)
        ) == {
            "alpha",
            "beta",
        }

    @pytest.mark.asyncio
    async def test_cross_timezone(self, properties_session):
        jst = timezone(timedelta(hours=9))
        cutoff = datetime(2024, 3, 15, 22, 0, 0, tzinfo=jst)  # = 13:00 UTC
        assert await _query_props_names(
            properties_session, Comparison("ts", ">=", cutoff)
        ) == {
            "beta",
            "gamma",
            "delta",
        }


class TestPropsJsonInFilters:
    @pytest.mark.asyncio
    async def test_int_in(self, properties_session):
        assert await _query_props_names(properties_session, In("count", [5, 15])) == {
            "alpha",
            "gamma",
        }

    @pytest.mark.asyncio
    async def test_string_in(self, properties_session):
        assert await _query_props_names(properties_session, In("tag", ["a", "b"])) == {
            "alpha",
            "beta",
        }

    @pytest.mark.asyncio
    async def test_empty_in(self, properties_session):
        assert await _query_props_names(properties_session, In("tag", [])) == set()


class TestPropsJsonIsNullFilter:
    @pytest.mark.asyncio
    async def test_is_null(self, properties_session):
        assert await _query_props_names(properties_session, IsNull("tag")) == {
            "epsilon"
        }


class TestPropsJsonLogicalFilters:
    @pytest.mark.asyncio
    async def test_and(self, properties_session):
        expr = And(Comparison("count", ">", 10), Comparison("active", "=", True))
        assert await _query_props_names(properties_session, expr) == {"gamma"}

    @pytest.mark.asyncio
    async def test_or(self, properties_session):
        expr = Or(Comparison("tag", "=", "a"), Comparison("tag", "=", "c"))
        assert await _query_props_names(properties_session, expr) == {"alpha", "gamma"}

    @pytest.mark.asyncio
    async def test_not(self, properties_session):
        expr = Not(Comparison("count", ">", 10))
        assert await _query_props_names(properties_session, expr) == {
            "alpha",
            "beta",
            "epsilon",
        }

    @pytest.mark.asyncio
    async def test_not_in(self, properties_session):
        expr = Not(In("tag", ["a", "b"]))
        assert await _query_props_names(properties_session, expr) == {"gamma", "delta"}

    @pytest.mark.asyncio
    async def test_compound_not_and(self, properties_session):
        inner = And(Comparison("count", ">", 10), Comparison("active", "=", True))
        assert await _query_props_names(properties_session, Not(inner)) == {
            "alpha",
            "beta",
            "delta",
            "epsilon",
        }


class TestPropsJsonCompileErrors:
    @pytest.mark.asyncio
    async def test_unsupported_operator(self, properties_session):
        with pytest.raises(ValueError, match="Unsupported operator"):
            await _query_props_names(
                properties_session, Comparison("count", cast(Any, "LIKE"), 10)
            )

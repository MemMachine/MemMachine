"""Integration tests for compile_sql_filter with an in-memory SQLite database.

These tests verify that JSON metadata filtering handles numeric types correctly
(integer ordering, float ordering, boolean equality) rather than falling back
to string/lexicographic comparison.
"""

import pytest
from sqlalchemy import JSON, Column, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Session

from memmachine.common.filter.filter_parser import parse_filter
from memmachine.common.filter.sql_filter_util import compile_sql_filter


class Base(DeclarativeBase):
    pass


class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    json_metadata = Column(JSON, nullable=True)


def _resolve_field(field: str):
    """Field resolver for the Item model."""
    normalized = field.lower()
    field_mapping = {
        "id": Item.id,
        "name": Item.name,
    }
    if normalized in field_mapping:
        return field_mapping[normalized], False

    if normalized.startswith(("p.", "properties.")):
        key = normalized.split(".", 1)[1]
        return Item.json_metadata[key], True

    return None, False


@pytest.fixture
def session():
    """Create an in-memory SQLite database with seeded rows."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        s.add_all(
            [
                Item(
                    name="alpha",
                    json_metadata={
                        "count": 5,
                        "score": 1.5,
                        "active": True,
                        "tag": "a",
                    },
                ),
                Item(
                    name="beta",
                    json_metadata={
                        "count": 10,
                        "score": 2.5,
                        "active": False,
                        "tag": "b",
                    },
                ),
                Item(
                    name="gamma",
                    json_metadata={
                        "count": 15,
                        "score": 3.5,
                        "active": True,
                        "tag": "c",
                    },
                ),
                Item(
                    name="delta",
                    json_metadata={
                        "count": 20,
                        "score": 4.5,
                        "active": False,
                        "tag": "d",
                    },
                ),
                Item(name="epsilon", json_metadata=None),
            ]
        )
        s.commit()
        yield s


def _query_names(session: Session, filter_str: str) -> set[str]:
    """Parse filter, compile to SQL, execute, and return set of matching names."""
    expr = parse_filter(filter_str)
    clause = compile_sql_filter(expr, _resolve_field)
    stmt = select(Item.name).where(clause)
    return {row[0] for row in session.execute(stmt)}


# --- Integer ordering (the core bug) ---


def test_int_greater_than(session):
    result = _query_names(session, "p.count > 10")
    assert result == {"gamma", "delta"}


def test_int_greater_equal(session):
    result = _query_names(session, "p.count >= 10")
    assert result == {"beta", "gamma", "delta"}


def test_int_less_than(session):
    result = _query_names(session, "p.count < 10")
    assert result == {"alpha"}


def test_int_less_equal(session):
    result = _query_names(session, "p.count <= 10")
    assert result == {"alpha", "beta"}


# --- Integer equality ---


def test_int_equality(session):
    result = _query_names(session, "p.count = 10")
    assert result == {"beta"}


def test_int_not_equal(session):
    result = _query_names(session, "p.count != 10")
    assert result == {"alpha", "gamma", "delta"}


# --- Float ordering ---


def test_float_greater_than(session):
    result = _query_names(session, "p.score > 2.0")
    assert result == {"beta", "gamma", "delta"}


def test_float_less_than(session):
    result = _query_names(session, "p.score < 3.0")
    assert result == {"alpha", "beta"}


def test_float_less_equal(session):
    result = _query_names(session, "p.score <= 2.5")
    assert result == {"alpha", "beta"}


# --- Boolean equality ---


def test_bool_equality_true(session):
    result = _query_names(session, "p.active = true")
    assert result == {"alpha", "gamma"}


def test_bool_equality_false(session):
    result = _query_names(session, "p.active = false")
    assert result == {"beta", "delta"}


# --- String equality ---


def test_string_equality(session):
    result = _query_names(session, "p.tag = 'a'")
    assert result == {"alpha"}


def test_string_not_equal(session):
    result = _query_names(session, "p.tag != 'a'")
    assert result == {"beta", "gamma", "delta"}


# --- IN with integers ---


def test_int_in(session):
    result = _query_names(session, "p.count IN (5, 15)")
    assert result == {"alpha", "gamma"}


# --- IN with strings ---


def test_string_in(session):
    result = _query_names(session, "p.tag IN ('a', 'b')")
    assert result == {"alpha", "beta"}


# --- IS NULL (missing metadata entirely) ---


def test_is_null(session):
    result = _query_names(session, "p.tag IS NULL")
    assert result == {"epsilon"}


# --- NOT / NOT IN ---


def test_not_comparison(session):
    result = _query_names(session, "NOT p.count > 10")
    # NOT (count > 10) => count <= 10 => alpha(5), beta(10)
    # epsilon has null metadata, so it won't match either side
    assert result == {"alpha", "beta"}


def test_not_in(session):
    result = _query_names(session, "p.tag NOT IN ('a', 'b')")
    assert result == {"gamma", "delta"}


# --- Non-metadata column (bypasses JSON casting) ---


def test_non_metadata_equality(session):
    result = _query_names(session, "name = 'alpha'")
    assert result == {"alpha"}


def test_non_metadata_in(session):
    result = _query_names(session, "name IN ('alpha', 'beta')")
    assert result == {"alpha", "beta"}


# --- NOT with And/Or compound expressions ---


def test_not_and_compound(session):
    # NOT (count > 10 AND active = true) => NOT(gamma, delta) => alpha, beta, epsilon
    # But only gamma satisfies count > 10 AND active = true
    # So NOT of that is: alpha, beta, delta, epsilon (since epsilon has null metadata)
    # Actually, gamma has count=15, active=True, so it matches inner.
    # Alpha has count=5, so doesn't match inner (count > 10 is false)
    # Beta has count=10 (10 > 10 is false), so doesn't match inner
    # Delta has count=20, active=False, so doesn't match inner (active != true)
    # Epsilon has null metadata, so doesn't match inner
    # So NOT of inner means: everything except gamma
    result = _query_names(session, "NOT (p.count > 10 AND p.active = true)")
    # Only gamma satisfies the inner condition (count=15 > 10, active=True)
    # So NOT gives us everyone except gamma
    assert result == {"alpha", "beta", "delta"}


def test_not_or_compound(session):
    # NOT (tag = 'a' OR tag = 'b') => NOT(alpha, beta) => gamma, delta
    result = _query_names(session, "NOT (p.tag = 'a' OR p.tag = 'b')")
    assert result == {"gamma", "delta"}

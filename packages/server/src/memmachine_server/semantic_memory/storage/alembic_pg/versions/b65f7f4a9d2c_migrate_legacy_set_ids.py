"""Migrate legacy set identifiers.

Revision ID: b65f7f4a9d2c
Revises: d1a9df11343b
Create Date: 2026-01-30 00:00:00.000000

"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision: str = "b65f7f4a9d2c"
down_revision: str | Sequence[str] | None = "d1a9df11343b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

USER_PREFIX = "mem_user_"
SESSION_PREFIX = "mem_session_"
ROLE_PREFIX = "mem_role_"


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _split_session_key(session_key: str, set_id: str) -> tuple[str, str | None]:
    cleaned = session_key.strip()
    if not cleaned:
        raise RuntimeError(f"Session key for {set_id} is empty")

    if "/" in cleaned:
        org_raw, project_raw = cleaned.split("/", 1)
        org_id = _clean(org_raw)
        project_id = _clean(project_raw)
    else:
        org_id = _clean(cleaned)
        project_id = None

    if org_id is None:
        raise RuntimeError(f"Unable to determine org_id for {set_id}")

    return org_id, project_id


def _hash_tag_list(strings: Iterable[str]) -> str:
    ordered = sorted(strings)
    hasher = hashlib.shake_256()
    for item in ordered:
        hasher.update(item.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest(6)


def _generate_set_id(
    *,
    org_id: str,
    project_id: str | None,
    metadata: Mapping[str, str],
) -> str:
    org_base = f"org_{org_id}"
    if project_id is not None:
        org_project = f"{org_base}_project_{project_id}"
        set_type = "project_set"
    else:
        org_project = org_base
        set_type = "set_type"

    metadata_keys = set(metadata.keys())
    if not metadata_keys:
        set_type = "project_set" if project_id is not None else "set_type"
    elif metadata_keys == {"producer_id"}:
        set_type = "user_set"
    else:
        set_type = "other_set"

    string_tags = [f"{key}_{metadata[key]}" for key in metadata]
    tag_hash = _hash_tag_list(metadata.keys())
    joined_tags = "_".join(sorted(string_tags))

    return f"mem_{set_type}_{org_project}_{len(metadata)}_{tag_hash}__{joined_tags}"


def _collect_candidate_set_ids(conn: sa.Connection) -> set[str]:
    inspector = inspect(conn)
    targets = (
        ("feature", "set_id"),
        ("set_ingested_history", "set_id"),
        ("semantic_config_setidresources", "set_id"),
        ("semantic_config_setidresources_settype", "set_id"),
        ("semantic_config_setidresources_disabledcategories", "set_id"),
        ("semantic_config_category", "set_id"),
    )

    candidates: set[str] = set()
    for table_name, column in targets:
        if not inspector.has_table(table_name):
            continue

        rows = conn.execute(
            sa.text(
                f"SELECT DISTINCT {column} FROM {table_name} WHERE {column} IS NOT NULL"
            )
        )

        for (set_id,) in rows:
            if not isinstance(set_id, str):
                continue
            cleaned = set_id.strip()
            if cleaned.startswith((USER_PREFIX, SESSION_PREFIX, ROLE_PREFIX)):
                candidates.add(cleaned)

    return candidates


def _build_migration_plan(conn: sa.Connection) -> dict[str, str]:
    candidate_set_ids = _collect_candidate_set_ids(conn)

    plan: dict[str, str] = {}
    for set_id in sorted(candidate_set_ids):
        if not set_id.startswith(SESSION_PREFIX):
            continue
        new_id = _translate_session_set_id(set_id)
        if new_id != set_id:
            plan[set_id] = new_id

    return plan


def _translate_session_set_id(set_id: str) -> str:
    raw_session_key = set_id[len(SESSION_PREFIX) :].strip()
    if not raw_session_key:
        raise RuntimeError(f"Unable to parse session key from {set_id}")

    org_id, project_id = _split_session_key(raw_session_key, set_id)

    return _generate_set_id(org_id=org_id, project_id=project_id, metadata={})


def _assert_no_config_collisions(
    conn: sa.Connection,
    plan: Mapping[str, str],
) -> None:
    inspector = inspect(conn)
    if not plan:
        return

    unique_tables = (
        "semantic_config_setidresources",
        "semantic_config_setidresources_settype",
        "semantic_config_setidresources_disabledcategories",
        "semantic_config_category",
    )

    for table_name in unique_tables:
        if not inspector.has_table(table_name):
            continue

        for old_id, new_id in plan.items():
            if old_id == new_id:
                continue
            existing = conn.execute(
                sa.text(f"SELECT 1 FROM {table_name} WHERE set_id = :set_id LIMIT 1"),
                {"set_id": new_id},
            ).first()
            if existing:
                raise RuntimeError(
                    f"Cannot migrate {old_id} to {new_id}: {table_name} already "
                    "contains the target set_id."
                )


def _apply_updates(
    conn: sa.Connection,
    *,
    plan: Mapping[str, str],
) -> None:
    if not plan:
        return

    _update_table(conn, "feature", "set_id", plan)
    _update_table(conn, "set_ingested_history", "set_id", plan)
    _update_table(conn, "semantic_config_setidresources", "set_id", plan)
    _update_table(conn, "semantic_config_setidresources_settype", "set_id", plan)
    _update_table(
        conn,
        "semantic_config_setidresources_disabledcategories",
        "set_id",
        plan,
    )
    _update_table(conn, "semantic_config_category", "set_id", plan)


def _update_table(
    conn: sa.Connection,
    table_name: str,
    column_name: str,
    plan: Mapping[str, str],
) -> None:
    if not plan or not inspect(conn).has_table(table_name):
        return

    stmt = sa.text(
        f"UPDATE {table_name} SET {column_name} = :new_id WHERE {column_name} = :old_id"
    )

    for old_id, new_id in plan.items():
        if old_id == new_id:
            continue
        conn.execute(stmt, {"old_id": old_id, "new_id": new_id})


def _migrate_legacy_set_ids(conn: sa.Connection) -> None:
    plan = _build_migration_plan(conn)
    _assert_no_config_collisions(conn, plan)
    _apply_updates(conn, plan=plan)


def upgrade() -> None:
    """Run the legacy set id migration."""
    conn = op.get_bind()
    _migrate_legacy_set_ids(conn)


def downgrade() -> None:
    """Downgrade is a no-op for this migration."""

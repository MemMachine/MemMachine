"""Add indexes and migrate legacy set identifiers.

Revision ID: d1a9df11343b
Revises: 62dff1150a46
Create Date: 2025-12-30 14:56:27.335328

"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import NamedTuple

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision: str = "d1a9df11343b"
down_revision: str | Sequence[str] | None = "62dff1150a46"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

USER_PREFIX = "mem_user_"
SESSION_PREFIX = "mem_session_"
ROLE_PREFIX = "mem_role_"


class ResolvedContext(NamedTuple):
    org_id: str | None
    project_id: str | None
    producer_id: str | None
    role_id: str | None
    session_key: str | None


class ContextAccumulator:
    __slots__ = ("producer_ids", "producer_roles", "session_keys")

    def __init__(self) -> None:
        self.session_keys: set[str] = set()
        self.producer_ids: set[str] = set()
        self.producer_roles: set[str] = set()

    def add(
        self,
        *,
        session_key: str | None,
        producer_id: str | None,
        producer_role: str | None,
    ) -> None:
        session_clean = _clean(session_key)
        if session_clean:
            self.session_keys.add(session_clean)

        producer_clean = _clean(producer_id)
        if producer_clean:
            self.producer_ids.add(producer_clean)

        role_clean = _clean(producer_role)
        if role_clean:
            self.producer_roles.add(role_clean)

    def resolve(self, *, set_id: str) -> ResolvedContext:
        session_key = _select_single(self.session_keys, "session_key", set_id)
        org_id: str | None = None
        project_id: str | None = None
        if session_key is not None:
            org_id, project_id = _split_session_key(session_key, set_id)

        producer_id = _select_single(self.producer_ids, "producer_id", set_id)
        role_id = _select_single(self.producer_roles, "producer_role", set_id)

        return ResolvedContext(
            org_id=org_id,
            project_id=project_id,
            producer_id=producer_id,
            role_id=role_id,
            session_key=session_key,
        )

    def is_empty(self) -> bool:
        return not (self.session_keys or self.producer_ids or self.producer_roles)


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _select_single(values: set[str], label: str, set_id: str) -> str | None:
    non_empty = {v for v in values if v}
    if len(non_empty) > 1:
        raise RuntimeError(
            f"Conflicting {label} values for {set_id}: {sorted(non_empty)}"
        )
    return next(iter(non_empty)) if non_empty else None


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


def _collect_contexts(
    conn: sa.Connection,
    *,
    set_ids: set[str],
) -> dict[str, ResolvedContext]:
    if not set_ids:
        return {}

    accumulators = {sid: ContextAccumulator() for sid in set_ids}

    history_links = _collect_history_links(conn, set_ids)
    citation_links = _collect_citation_links(conn, set_ids)

    all_history_ids = set(history_links.keys()) | set(citation_links.keys())
    episodes = _fetch_episode_rows(conn, all_history_ids)

    for history_id, parents in history_links.items():
        episode = episodes.get(history_id)
        if episode is None:
            continue
        for set_id in parents:
            accumulators[set_id].add(
                session_key=episode["session_key"],
                producer_id=episode["producer_id"],
                producer_role=episode["producer_role"],
            )

    for history_id, parents in citation_links.items():
        episode = episodes.get(history_id)
        if episode is None:
            continue
        for set_id in parents:
            accumulators[set_id].add(
                session_key=episode["session_key"],
                producer_id=episode["producer_id"],
                producer_role=episode["producer_role"],
            )

    resolved: dict[str, ResolvedContext] = {}
    for set_id, accumulator in accumulators.items():
        if accumulator.is_empty():
            continue
        resolved[set_id] = accumulator.resolve(set_id=set_id)

    return resolved


def _collect_history_links(
    conn: sa.Connection,
    set_ids: set[str],
) -> dict[int, list[str]]:
    if not set_ids or not inspect(conn).has_table("set_ingested_history"):
        return {}

    table = sa.table(
        "set_ingested_history",
        sa.column("set_id", sa.String),
        sa.column("history_id", sa.String),
    )

    stmt = sa.select(table.c.set_id, table.c.history_id).where(
        table.c.set_id.in_(tuple(set_ids))
    )

    mapping: dict[int, list[str]] = {}
    for set_id, history_id in conn.execute(stmt):
        if history_id is None:
            continue
        history_str = str(history_id)
        if not history_str.isdigit():
            continue
        history_int = int(history_str)
        mapping.setdefault(history_int, []).append(set_id)

    return mapping


def _collect_citation_links(
    conn: sa.Connection,
    set_ids: set[str],
) -> dict[int, list[str]]:
    inspector = inspect(conn)
    if (
        not set_ids
        or not inspector.has_table("citations")
        or not inspector.has_table("feature")
    ):
        return {}

    feature_table = sa.table(
        "feature",
        sa.column("id", sa.Integer),
        sa.column("set_id", sa.String),
    )
    citations_table = sa.table(
        "citations",
        sa.column("feature_id", sa.Integer),
        sa.column("history_id", sa.String),
    )

    stmt = (
        sa.select(feature_table.c.set_id, citations_table.c.history_id)
        .join(citations_table, feature_table.c.id == citations_table.c.feature_id)
        .where(feature_table.c.set_id.in_(tuple(set_ids)))
    )

    mapping: dict[int, list[str]] = {}
    for set_id, history_id in conn.execute(stmt):
        if history_id is None:
            continue
        history_str = str(history_id)
        if not history_str.isdigit():
            continue
        history_int = int(history_str)
        mapping.setdefault(history_int, []).append(set_id)

    return mapping


def _fetch_episode_rows(
    conn: sa.Connection,
    history_ids: set[int],
) -> dict[int, dict[str, str | None]]:
    inspector = inspect(conn)
    if not history_ids or not inspector.has_table("episodestore"):
        raise RuntimeError(
            "episodestore table is required to migrate legacy set identifiers"
        )

    episodes_table = sa.table(
        "episodestore",
        sa.column("id", sa.BigInteger),
        sa.column("session_key", sa.String),
        sa.column("producer_id", sa.String),
        sa.column("producer_role", sa.String),
    )

    results: dict[int, dict[str, str | None]] = {}
    id_list = list(history_ids)
    chunk_size = 1000

    for start in range(0, len(id_list), chunk_size):
        chunk = id_list[start : start + chunk_size]
        stmt = sa.select(episodes_table).where(episodes_table.c.id.in_(chunk))
        for row in conn.execute(stmt):
            history_id = int(row.id)  # type: ignore[union-attr]
            results[history_id] = {
                "session_key": row.session_key,  # type: ignore[union-attr]
                "producer_id": row.producer_id,  # type: ignore[union-attr]
                "producer_role": row.producer_role,  # type: ignore[union-attr]
            }

    return results


def _build_migration_plan(
    conn: sa.Connection,
) -> tuple[dict[str, str], dict[str, ResolvedContext]]:
    candidate_set_ids = _collect_candidate_set_ids(conn)
    contexts = _collect_contexts(conn, set_ids=candidate_set_ids)

    plan: dict[str, str] = {}
    for set_id in sorted(candidate_set_ids):
        if set_id.startswith(SESSION_PREFIX):
            new_id = _translate_session_set_id(set_id, contexts.get(set_id))
        elif set_id.startswith(USER_PREFIX):
            context = contexts.get(set_id)
            if context is None or context.org_id is None or context.producer_id is None:
                raise RuntimeError(
                    "Unable to resolve org/project/producer for legacy user set "
                    f"{set_id}. Ensure episodic history exists for these entries."
                )
            new_id = _generate_set_id(
                org_id=context.org_id,
                project_id=context.project_id,
                metadata={"producer_id": context.producer_id},
            )
        elif set_id.startswith(ROLE_PREFIX):
            context = contexts.get(set_id)
            if context is None or context.org_id is None:
                raise RuntimeError(
                    f"Unable to resolve org/project for legacy role set {set_id}."
                )
            role_identifier = context.role_id or set_id[len(ROLE_PREFIX) :].strip()
            if not role_identifier:
                raise RuntimeError(f"Unable to determine role identifier for {set_id}")
            new_id = _generate_set_id(
                org_id=context.org_id,
                project_id=context.project_id,
                metadata={"role_id": role_identifier},
            )
        else:
            continue

        if new_id != set_id:
            plan[set_id] = new_id

    return plan, contexts


def _translate_session_set_id(
    set_id: str,
    context: ResolvedContext | None,
) -> str:
    raw_session_key = set_id[len(SESSION_PREFIX) :].strip()
    if not raw_session_key:
        raise RuntimeError(f"Unable to parse session key from {set_id}")

    org_id, project_id = _split_session_key(raw_session_key, set_id)

    if context is not None:
        if context.org_id is not None and context.org_id != org_id:
            raise RuntimeError(
                f"Session key mismatch for {set_id}: {context.org_id} != {org_id}"
            )
        if context.project_id is not None and context.project_id != project_id:
            raise RuntimeError(
                f"Session project mismatch for {set_id}: "
                f"{context.project_id} != {project_id}"
            )

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
    contexts: Mapping[str, ResolvedContext],
) -> None:
    if not plan:
        return

    _update_table(conn, "feature", "set_id", plan)
    _update_table(conn, "set_ingested_history", "set_id", plan)
    _update_table(conn, "semantic_config_setidresources", "set_id", plan)
    _update_config_settype(conn, plan=plan, contexts=contexts)
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


def _update_config_settype(
    conn: sa.Connection,
    *,
    plan: Mapping[str, str],
    contexts: Mapping[str, ResolvedContext],
) -> None:
    table_name = "semantic_config_setidresources_settype"
    if not inspect(conn).has_table(table_name):
        return

    base_stmt = sa.text(
        "UPDATE semantic_config_setidresources_settype "
        "SET set_id = :new_id WHERE set_id = :old_id"
    )

    specialized_stmt = sa.text(
        "UPDATE semantic_config_setidresources_settype "
        "SET set_id = :new_id, set_type_id = :set_type_id "
        "WHERE set_id = :old_id"
    )

    for old_id, new_id in plan.items():
        if old_id == new_id:
            continue

        if old_id.startswith(USER_PREFIX):
            context = contexts.get(old_id)
            if context is None or context.org_id is None:
                raise RuntimeError(
                    f"Missing context for user set {old_id} during config migration"
                )
            set_type_id = _ensure_user_set_type(conn, context.org_id)
            conn.execute(
                specialized_stmt,
                {
                    "old_id": old_id,
                    "new_id": new_id,
                    "set_type_id": set_type_id,
                },
            )
        else:
            conn.execute(base_stmt, {"old_id": old_id, "new_id": new_id})


def _ensure_user_set_type(conn: sa.Connection, org_id: str) -> int:
    stmt = sa.text(
        "SELECT id FROM set_type "
        "WHERE org_id = :org_id AND org_level_set = FALSE "
        "AND metadata_tags_sig = :sig"
    )

    existing = conn.execute(stmt, {"org_id": org_id, "sig": "producer_id"}).first()
    if existing:
        return int(existing[0])

    insert_stmt = sa.text(
        "INSERT INTO set_type (org_id, org_level_set, metadata_tags_sig, name, description) "
        "VALUES (:org_id, FALSE, :sig, :name, :description) RETURNING id"
    )

    result = conn.execute(
        insert_stmt,
        {
            "org_id": org_id,
            "sig": "producer_id",
            "name": "User Profile",
            "description": "Semantic memory scoped to producer identifiers.",
        },
    )
    return int(result.scalar_one())


def _migrate_legacy_set_ids(conn: sa.Connection) -> None:
    plan, contexts = _build_migration_plan(conn)
    _assert_no_config_collisions(conn, plan)
    _apply_updates(conn, plan=plan, contexts=contexts)


def upgrade() -> None:
    conn = op.get_bind()
    _migrate_legacy_set_ids(conn)

    op.create_index(op.f("ix_feature_set_id"), "feature", ["set_id"], unique=False)
    op.create_index(
        op.f("ix_set_ingested_history_set_id"),
        "set_ingested_history",
        ["set_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_set_ingested_history_set_id"), table_name="set_ingested_history"
    )
    op.drop_index(op.f("ix_feature_set_id"), table_name="feature")

"""Neo4j migration utilities for knowledge-graph improvements.

This module provides one-time migration functions to prepare an existing
Neo4j database for the MERGE-based upsert and entity-type-label features
introduced in the knowledge-graph improvement change.

Typical usage::

    from neo4j import AsyncGraphDatabase
    from memmachine.common.vector_graph_store.neo4j_migration import (
        audit_duplicate_uids,
        resolve_duplicate_uids,
        apply_uniqueness_constraints,
        backfill_entity_type_labels,
    )

    driver = AsyncGraphDatabase.driver(...)
    report = await audit_duplicate_uids(driver)
    await resolve_duplicate_uids(driver)
    await apply_uniqueness_constraints(driver)
    await backfill_entity_type_labels(driver, {"memories": {"Person": [...]}})
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import LiteralString, cast

from neo4j import AsyncDriver, AsyncTransaction, Query
from neo4j.graph import Node as Neo4jNode

from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
)

logger = logging.getLogger(__name__)


def _neo4j_query(text: str) -> Query:
    return Query(cast(LiteralString, text))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SANITIZE_PREFIX = Neo4jVectorGraphStore._SANITIZE_NAME_PREFIX  # noqa: SLF001


async def _get_collection_labels(driver: AsyncDriver) -> list[str]:
    """Return all Neo4j labels that represent collections (SANITIZED_ prefix)."""
    records, _, _ = await driver.execute_query(
        _neo4j_query("CALL db.labels() YIELD label RETURN label"),
    )
    return [
        record["label"]
        for record in records
        if record["label"].startswith(_SANITIZE_PREFIX)
    ]


# ---------------------------------------------------------------------------
# 3.2 -- audit_duplicate_uids
# ---------------------------------------------------------------------------


@dataclass
class DuplicateReport:
    """Summary of duplicate UIDs across all collections."""

    total_duplicates: int = 0
    by_collection: dict[str, list[DuplicateInfo]] = field(default_factory=dict)


@dataclass
class DuplicateInfo:
    """Details about a single duplicate UID within a collection."""

    uid: str
    count: int


async def audit_duplicate_uids(driver: AsyncDriver) -> DuplicateReport:
    """Query each collection for duplicate UIDs and log counts.

    This is a **read-only** operation -- no data is modified.

    Returns:
        A ``DuplicateReport`` summarising duplicates per collection.
    """
    report = DuplicateReport()
    collection_labels = await _get_collection_labels(driver)

    for label in collection_labels:
        records, _, _ = await driver.execute_query(
            _neo4j_query(
                f"MATCH (n:{label})\n"
                "WITH n.uid AS uid, count(*) AS cnt\n"
                "WHERE cnt > 1\n"
                "RETURN uid, cnt\n"
                "ORDER BY cnt DESC"
            ),
        )

        if records:
            duplicates = [DuplicateInfo(uid=r["uid"], count=r["cnt"]) for r in records]
            desanitized = Neo4jVectorGraphStore._desanitize_name(label)  # noqa: SLF001
            report.by_collection[desanitized] = duplicates
            total = sum(d.count - 1 for d in duplicates)  # surplus nodes
            report.total_duplicates += total
            logger.info(
                "audit_duplicate_uids: collection %r has %d duplicate UID(s) "
                "across %d unique UIDs",
                desanitized,
                total,
                len(duplicates),
            )

    if report.total_duplicates == 0:
        logger.info("audit_duplicate_uids: no duplicates found")

    return report


# ---------------------------------------------------------------------------
# 3.3 -- resolve_duplicate_uids
# ---------------------------------------------------------------------------


def _pick_canonical_node(nodes: list[Neo4jNode]) -> Neo4jNode:
    """Return the node with the most properties (heuristic for richest data)."""
    canonical = nodes[0]
    for node in nodes[1:]:
        if len(dict(node.items())) > len(dict(canonical.items())):
            canonical = node
    return canonical


async def _merge_surplus_node(
    tx: AsyncTransaction, canonical: Neo4jNode, surplus_node: Neo4jNode
) -> None:
    """Merge a single surplus node into the canonical node within *tx*.

    Steps: merge properties, repoint incoming & outgoing relationships,
    then delete the surplus node.
    """
    cid = canonical.element_id
    sid = surplus_node.element_id

    # Merge properties onto canonical.
    await tx.run(
        _neo4j_query(
            "MATCH (canonical) WHERE elementId(canonical) = $cid\n"
            "MATCH (surplus) WHERE elementId(surplus) = $sid\n"
            "SET canonical += properties(surplus)"
        ),
        cid=cid,
        sid=sid,
    )

    # Repoint incoming relationships.
    await tx.run(
        _neo4j_query(
            "MATCH (surplus) WHERE elementId(surplus) = $sid\n"
            "MATCH (canonical) WHERE elementId(canonical) = $cid\n"
            "MATCH (other)-[r]->(surplus)\n"
            "WITH other, r, canonical, type(r) AS rtype, "
            "properties(r) AS rprops\n"
            "CALL apoc.create.relationship("
            "  other, rtype, rprops, canonical"
            ") YIELD rel\n"
            "DELETE r"
        ),
        sid=sid,
        cid=cid,
    )

    # Repoint outgoing relationships.
    await tx.run(
        _neo4j_query(
            "MATCH (surplus) WHERE elementId(surplus) = $sid\n"
            "MATCH (canonical) WHERE elementId(canonical) = $cid\n"
            "MATCH (surplus)-[r]->(other)\n"
            "WITH other, r, canonical, type(r) AS rtype, "
            "properties(r) AS rprops\n"
            "CALL apoc.create.relationship("
            "  canonical, rtype, rprops, other"
            ") YIELD rel\n"
            "DELETE r"
        ),
        sid=sid,
        cid=cid,
    )

    # Delete surplus node.
    await tx.run(
        _neo4j_query(
            "MATCH (surplus) WHERE elementId(surplus) = $sid\nDETACH DELETE surplus"
        ),
        sid=sid,
    )


async def resolve_duplicate_uids(driver: AsyncDriver) -> int:
    """Resolve duplicate UIDs in every collection.

    For each duplicate group the function:

    1. Picks the canonical node (the one with the most properties as a
       heuristic for "most recent / richest data").
    2. Merges properties from surplus nodes onto the canonical node
       (``SET canonical += surplus.properties`` so newer keys overwrite).
    3. Repoints all relationships from surplus nodes to the canonical node
       using ``apoc.create.relationship``.
    4. Deletes surplus nodes.

    All steps for a single collection run inside a single explicit
    transaction so the operation is atomic per collection.

    Returns:
        The total number of surplus nodes deleted across all collections.
    """
    collection_labels = await _get_collection_labels(driver)
    total_deleted = 0

    for label in collection_labels:
        deleted = await _resolve_duplicates_for_label(driver, label)
        total_deleted += deleted

    logger.info(
        "resolve_duplicate_uids: deleted %d surplus node(s) total",
        total_deleted,
    )
    return total_deleted


async def _resolve_duplicates_for_label(driver: AsyncDriver, label: str) -> int:
    """Resolve all duplicate UID groups for a single collection label.

    Returns the number of surplus nodes deleted.
    """
    records, _, _ = await driver.execute_query(
        _neo4j_query(
            f"MATCH (n:{label})\n"
            "WITH n.uid AS uid, collect(n) AS nodes, count(*) AS cnt\n"
            "WHERE cnt > 1\n"
            "RETURN uid, nodes"
        ),
    )

    if not records:
        return 0

    desanitized = Neo4jVectorGraphStore._desanitize_name(label)  # noqa: SLF001
    logger.info(
        "resolve_duplicate_uids: resolving %d duplicate group(s) in collection %r",
        len(records),
        desanitized,
    )

    deleted = 0
    async with driver.session() as session:
        tx = await session.begin_transaction()
        try:
            for record in records:
                uid: str = record["uid"]
                nodes: list[Neo4jNode] = list(record["nodes"])

                if len(nodes) < 2:
                    continue  # pragma: no cover -- safety guard

                canonical = _pick_canonical_node(nodes)
                surplus = [n for n in nodes if n.element_id != canonical.element_id]

                for surplus_node in surplus:
                    await _merge_surplus_node(tx, canonical, surplus_node)
                    deleted += 1
                    logger.debug(
                        "resolve_duplicate_uids: deleted surplus node "
                        "for uid=%r in collection %r",
                        uid,
                        desanitized,
                    )

            await tx.commit()
        except Exception:
            await tx.rollback()
            raise

    return deleted


# ---------------------------------------------------------------------------
# 3.4 -- apply_uniqueness_constraints
# ---------------------------------------------------------------------------


async def apply_uniqueness_constraints(driver: AsyncDriver) -> list[str]:
    """Create uid uniqueness constraints for all active collection labels.

    Uses ``CREATE CONSTRAINT ... IF NOT EXISTS`` so this is safe to call
    multiple times (idempotent).

    Returns:
        List of constraint names that were ensured.
    """
    collection_labels = await _get_collection_labels(driver)
    constraint_names: list[str] = []

    for label in collection_labels:
        constraint_name = f"unique_uid_{label}"
        await driver.execute_query(
            _neo4j_query(
                f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.uid IS UNIQUE"
            ),
        )
        constraint_names.append(constraint_name)
        logger.info(
            "apply_uniqueness_constraints: ensured constraint %s",
            constraint_name,
        )

    return constraint_names


# ---------------------------------------------------------------------------
# 3.5 -- backfill_entity_type_labels
# ---------------------------------------------------------------------------


async def backfill_entity_type_labels(
    driver: AsyncDriver,
    mapping: dict[str, dict[str, list[str]]] | None = None,
) -> int:
    """Apply entity type labels to existing nodes based on a mapping.

    *mapping* structure::

        {
            "collection_name": {
                "EntityType": ["uid1", "uid2", ...],
                ...
            },
            ...
        }

    If *mapping* is ``None`` or empty this is a no-op (forward-only typing).

    Returns:
        Number of nodes updated.
    """
    if not mapping:
        logger.info("backfill_entity_type_labels: no mapping provided, skipping")
        return 0

    total_updated = 0

    for collection_name, type_uids in mapping.items():
        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(  # noqa: SLF001
            collection_name,
        )

        for entity_type, uids in type_uids.items():
            if not uids:
                continue

            sanitized_type = Neo4jVectorGraphStore._sanitize_entity_type(  # noqa: SLF001
                entity_type,
            )
            result = await driver.execute_query(
                _neo4j_query(
                    "UNWIND $uids AS uid\n"
                    f"MATCH (n:{sanitized_collection} {{uid: uid}})\n"
                    f"SET n:{sanitized_type}\n"
                    "RETURN count(n) AS updated"
                ),
                uids=uids,
            )
            count = result[0][0]["updated"] if result[0] else 0
            total_updated += count
            logger.info(
                "backfill_entity_type_labels: applied %r to %d node(s) "
                "in collection %r",
                entity_type,
                count,
                collection_name,
            )

    logger.info(
        "backfill_entity_type_labels: updated %d node(s) total",
        total_updated,
    )
    return total_updated

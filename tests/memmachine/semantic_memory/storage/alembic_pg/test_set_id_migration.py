from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine.semantic_memory.semantic_session_manager import SemanticSessionManager
from memmachine.semantic_memory.storage import (
    sqlalchemy_pgvector_semantic as storage_mod,
)
from memmachine.semantic_memory.storage.sqlalchemy_pgvector_semantic import (
    BaseSemanticStorage,
)

pytestmark = pytest.mark.integration

_SCRIPT_LOCATION = Path(storage_mod.__file__).parent / "alembic_pg"
_VERSIONS_LOCATION = _SCRIPT_LOCATION / "versions"


async def _reset_database(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:  # pragma: no cover - helper
        await conn.run_sync(
            lambda sync_conn: BaseSemanticStorage.metadata.drop_all(bind=sync_conn)
        )
        for table in (
            "semantic_config_category",
            "semantic_config_setidresources_disabledcategories",
            "semantic_config_setidresources_settype",
            "semantic_config_setidresources",
            "set_type",
            "episodestore",
            "alembic_version",
        ):
            await conn.execute(sa.text(f"DROP TABLE IF EXISTS {table} CASCADE"))


async def _run_upgrade(engine: AsyncEngine, target: str) -> None:
    async with engine.begin() as conn:  # pragma: no cover - helper

        def _upgrade(sync_conn):
            config = Config()
            config.set_main_option("script_location", str(_SCRIPT_LOCATION))
            config.set_main_option("version_locations", str(_VERSIONS_LOCATION))
            config.set_main_option("path_separator", "os")
            config.set_main_option("sqlalchemy.url", str(sync_conn.engine.url))
            config.attributes["connection"] = sync_conn
            command.upgrade(config, target)

        await conn.run_sync(_upgrade)


async def _fetch_migration_state(
    engine: AsyncEngine,
) -> tuple[
    set[str],
    set[str],
]:
    async with engine.connect() as conn:
        feature_rows = await conn.execute(sa.text("SELECT set_id FROM feature"))
        feature_set_ids = {row[0] for row in feature_rows}

        history_rows = await conn.execute(
            sa.text("SELECT set_id FROM set_ingested_history")
        )
        history_set_ids = {row[0] for row in history_rows}

    return (
        feature_set_ids,
        history_set_ids,
    )


def _split_session_key_for_test(session_key: str) -> tuple[str, str | None]:
    cleaned = session_key.strip()
    if "/" in cleaned:
        org_id, project_id = cleaned.split("/", 1)
    else:
        org_id, project_id = cleaned, None
    return org_id, project_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "legacy_set_id",
    [
        "mem_session_acme/project-x",
        "mem_session_universal/project_6fdqmczu_empty_fields",
    ],
)
async def test_legacy_set_ids_are_transformed(
    sqlalchemy_pg_engine: AsyncEngine,
    legacy_set_id: str,
):
    await _reset_database(sqlalchemy_pg_engine)

    await _run_upgrade(sqlalchemy_pg_engine, "d1a9df11343b")

    session_key = legacy_set_id.removeprefix("mem_session_").strip()
    org_id, project_id = _split_session_key_for_test(session_key)

    async with sqlalchemy_pg_engine.begin() as conn:
        await conn.execute(
            sa.text(
                "INSERT INTO feature "
                "(set_id, semantic_category_id, tag_id, feature, value) "
                "VALUES (:set_id, 'profile', 'tag', 'topic', 'value')"
            ),
            {"set_id": legacy_set_id},
        )

        await conn.execute(
            sa.text(
                "INSERT INTO set_ingested_history (set_id, history_id, ingested) "
                "VALUES (:set_id, :history_id, FALSE)"
            ),
            {"set_id": legacy_set_id, "history_id": "101"},
        )

    await _run_upgrade(sqlalchemy_pg_engine, "head")

    expected_set_id = SemanticSessionManager._generate_set_id(
        org_id=org_id,
        project_id=project_id,
        metadata={},
    )

    feature_set_ids, history_set_ids = await _fetch_migration_state(
        sqlalchemy_pg_engine
    )

    assert expected_set_id in feature_set_ids
    assert expected_set_id in history_set_ids
    assert legacy_set_id not in feature_set_ids
    assert legacy_set_id not in history_set_ids

    await _reset_database(sqlalchemy_pg_engine)


@pytest.mark.asyncio
async def test_user_and_role_set_ids_are_unchanged(
    sqlalchemy_pg_engine: AsyncEngine,
) -> None:
    await _reset_database(sqlalchemy_pg_engine)

    await _run_upgrade(sqlalchemy_pg_engine, "d1a9df11343b")

    user_set_id = "mem_user_legacy"
    role_set_id = "mem_role_operator"

    async with sqlalchemy_pg_engine.begin() as conn:
        for legacy_id in (user_set_id, role_set_id):
            await conn.execute(
                sa.text(
                    "INSERT INTO feature (set_id, semantic_category_id, tag_id, feature, value) "
                    "VALUES (:set_id, 'profile', 'tag', 'topic', 'value')"
                ),
                {"set_id": legacy_id},
            )

            await conn.execute(
                sa.text(
                    "INSERT INTO set_ingested_history (set_id, history_id, ingested) "
                    "VALUES (:set_id, :history_id, FALSE)"
                ),
                {"set_id": legacy_id, "history_id": "101"},
            )

    await _run_upgrade(sqlalchemy_pg_engine, "head")

    feature_set_ids, history_set_ids = await _fetch_migration_state(
        sqlalchemy_pg_engine
    )

    for legacy_id in (user_set_id, role_set_id):
        assert legacy_id in feature_set_ids
        assert legacy_id in history_set_ids

    await _reset_database(sqlalchemy_pg_engine)

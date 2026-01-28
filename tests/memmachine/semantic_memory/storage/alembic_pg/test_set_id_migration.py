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


async def _create_supporting_tables(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            sa.text(
                """
                CREATE TABLE episodestore (
                    id BIGINT PRIMARY KEY,
                    session_key TEXT NOT NULL,
                    producer_id TEXT NOT NULL,
                    producer_role TEXT
                )
                """
            )
        )

        await conn.execute(
            sa.text(
                """
                CREATE TABLE IF NOT EXISTS set_type (
                    id SERIAL PRIMARY KEY,
                    org_id TEXT NOT NULL,
                    org_level_set BOOLEAN NOT NULL,
                    metadata_tags_sig TEXT NOT NULL,
                    name TEXT,
                    description TEXT
                )
                """
            )
        )

        await conn.execute(
            sa.text(
                """
                CREATE TABLE IF NOT EXISTS semantic_config_setidresources (
                    set_id TEXT PRIMARY KEY
                )
                """
            )
        )

        await conn.execute(
            sa.text(
                """
                CREATE TABLE IF NOT EXISTS semantic_config_setidresources_settype (
                    set_id TEXT PRIMARY KEY,
                    set_type_id INTEGER NOT NULL REFERENCES set_type(id)
                )
                """
            )
        )

        await conn.execute(
            sa.text(
                """
                CREATE TABLE IF NOT EXISTS semantic_config_setidresources_disabledcategories (
                    set_id TEXT NOT NULL,
                    disabled_category TEXT NOT NULL,
                    PRIMARY KEY (set_id, disabled_category)
                )
                """
            )
        )

        await conn.execute(
            sa.text(
                """
                CREATE TABLE IF NOT EXISTS semantic_config_category (
                    id SERIAL PRIMARY KEY,
                    set_id TEXT,
                    name TEXT NOT NULL,
                    prompt TEXT NOT NULL
                )
                """
            )
        )


@pytest.mark.asyncio
async def test_legacy_set_ids_are_transformed(sqlalchemy_pg_engine: AsyncEngine):
    await _reset_database(sqlalchemy_pg_engine)

    await _run_upgrade(sqlalchemy_pg_engine, "62dff1150a46")

    await _create_supporting_tables(sqlalchemy_pg_engine)

    legacy_session_id = "mem_session_acme/project-x"
    legacy_user_id = "mem_user_alice"
    legacy_role_id = "mem_role_admin"
    untouched_set_id = "custom-set"

    async with sqlalchemy_pg_engine.begin() as conn:
        await conn.execute(
            sa.text(
                "INSERT INTO episodestore (id, session_key, producer_id, producer_role) "
                "VALUES (:id, :session_key, :producer_id, :producer_role)"
            ),
            {
                "id": 101,
                "session_key": "acme/project-x",
                "producer_id": "alice",
                "producer_role": "admin",
            },
        )

        await conn.execute(
            sa.text(
                "INSERT INTO set_type (org_id, org_level_set, metadata_tags_sig, name, description) "
                "VALUES ('acme', FALSE, 'legacy', 'Legacy Type', 'Legacy mapping')"
            )
        )

        await conn.execute(
            sa.text(
                "INSERT INTO semantic_config_setidresources (set_id) VALUES (:set_id)"
            ),
            {"set_id": legacy_user_id},
        )

        await conn.execute(
            sa.text(
                "INSERT INTO semantic_config_setidresources_settype (set_id, set_type_id) "
                "VALUES (:set_id, 1)"
            ),
            {"set_id": legacy_user_id},
        )

        await conn.execute(
            sa.text(
                "INSERT INTO semantic_config_setidresources_disabledcategories (set_id, disabled_category) "
                "VALUES (:set_id, 'legacy-cat')"
            ),
            {"set_id": legacy_user_id},
        )

        await conn.execute(
            sa.text(
                "INSERT INTO semantic_config_category (set_id, name, prompt) "
                "VALUES (:set_id, 'legacy-category', 'prompt')"
            ),
            {"set_id": legacy_user_id},
        )

        await conn.execute(
            sa.text(
                "INSERT INTO feature "
                "(set_id, semantic_category_id, tag_id, feature, value) "
                "VALUES (:set_id, 'profile', 'tag', 'topic', 'value')"
            ),
            {"set_id": legacy_session_id},
        )
        await conn.execute(
            sa.text(
                "INSERT INTO feature "
                "(set_id, semantic_category_id, tag_id, feature, value) "
                "VALUES (:set_id, 'profile', 'tag', 'topic', 'other')"
            ),
            {"set_id": untouched_set_id},
        )
        await conn.execute(
            sa.text(
                "INSERT INTO set_ingested_history (set_id, history_id, ingested) "
                "VALUES (:set_id, :history_id, FALSE)"
            ),
            {"set_id": legacy_session_id, "history_id": "legacy-history"},
        )
        await conn.execute(
            sa.text(
                "INSERT INTO set_ingested_history (set_id, history_id, ingested) "
                "VALUES (:set_id, :history_id, FALSE)"
            ),
            {"set_id": untouched_set_id, "history_id": "other-history"},
        )

        await conn.execute(
            sa.text(
                "INSERT INTO feature (set_id, semantic_category_id, tag_id, feature, value) "
                "VALUES (:set_id, 'profile', 'tag', 'topic', 'user-value')"
            ),
            {"set_id": legacy_user_id},
        )

        await conn.execute(
            sa.text(
                "INSERT INTO feature (set_id, semantic_category_id, tag_id, feature, value) "
                "VALUES (:set_id, 'profile', 'tag', 'topic', 'role-value')"
            ),
            {"set_id": legacy_role_id},
        )

        await conn.execute(
            sa.text(
                "INSERT INTO set_ingested_history (set_id, history_id, ingested) "
                "VALUES (:set_id, :history_id, FALSE)"
            ),
            {"set_id": legacy_user_id, "history_id": "101"},
        )

        await conn.execute(
            sa.text(
                "INSERT INTO set_ingested_history (set_id, history_id, ingested) "
                "VALUES (:set_id, :history_id, FALSE)"
            ),
            {"set_id": legacy_role_id, "history_id": "101"},
        )

    await _run_upgrade(sqlalchemy_pg_engine, "head")

    expected_project_set_id = SemanticSessionManager._generate_set_id(
        org_id="acme",
        project_id="project-x",
        metadata={},
    )

    expected_user_set_id = SemanticSessionManager.generate_user_set_id(
        org_id="acme",
        project_id="project-x",
        producer_id="alice",
    )

    expected_role_set_id = SemanticSessionManager._generate_set_id(
        org_id="acme",
        project_id="project-x",
        metadata={"role_id": "admin"},
    )

    async with sqlalchemy_pg_engine.connect() as conn:
        feature_rows = await conn.execute(sa.text("SELECT set_id FROM feature"))
        feature_set_ids = {row[0] for row in feature_rows}

        history_rows = await conn.execute(
            sa.text("SELECT set_id FROM set_ingested_history")
        )
        history_set_ids = {row[0] for row in history_rows}

        config_rows = await conn.execute(
            sa.text("SELECT set_id FROM semantic_config_setidresources")
        )
        config_set_ids = {row[0] for row in config_rows}

        mapping_rows = await conn.execute(
            sa.text(
                "SELECT set_id, set_type_id FROM semantic_config_setidresources_settype"
            )
        )
        mapping_data = [(row[0], row[1]) for row in mapping_rows]

        set_type_rows = await conn.execute(
            sa.text("SELECT id, metadata_tags_sig FROM set_type")
        )
        set_type_by_sig = {row[1]: row[0] for row in set_type_rows}

        disabled_rows = await conn.execute(
            sa.text(
                "SELECT set_id FROM semantic_config_setidresources_disabledcategories"
            )
        )
        disabled_set_ids = {row[0] for row in disabled_rows}

        category_rows = await conn.execute(
            sa.text("SELECT set_id FROM semantic_config_category")
        )
        category_set_ids = {row[0] for row in category_rows}

    assert expected_project_set_id in feature_set_ids
    assert expected_user_set_id in feature_set_ids
    assert expected_role_set_id in feature_set_ids
    assert untouched_set_id in feature_set_ids
    assert expected_project_set_id in history_set_ids
    assert expected_user_set_id in history_set_ids
    assert expected_role_set_id in history_set_ids
    assert untouched_set_id in history_set_ids
    assert legacy_session_id not in feature_set_ids
    assert legacy_user_id not in feature_set_ids
    assert legacy_role_id not in feature_set_ids
    assert legacy_session_id not in history_set_ids
    assert legacy_user_id not in history_set_ids
    assert legacy_role_id not in history_set_ids

    assert expected_user_set_id in config_set_ids

    assert expected_user_set_id in disabled_set_ids
    assert expected_user_set_id in category_set_ids

    user_set_type_id = set_type_by_sig.get("producer_id")
    assert user_set_type_id is not None
    assert (expected_user_set_id, user_set_type_id) in mapping_data

    assert legacy_user_id not in config_set_ids

    await _reset_database(sqlalchemy_pg_engine)

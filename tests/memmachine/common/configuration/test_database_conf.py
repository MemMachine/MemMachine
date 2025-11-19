import pytest
from pydantic import SecretStr

from memmachine.common.configuration.database_conf import (
    DatabasesConf,
    Neo4jConf,
    SqlAlchemyConf,
)


def test_parse_valid_storage_dict():
    input_dict = {
        "databases": {
            "my_neo4j": {
                "provider": "neo4j",
                "config": {
                    "host": "localhost",
                    "port": 7687,
                    "user": "neo4j",
                    "password": "secret",
                },
            },
            "main_postgres": {
                "provider": "postgres",
                "config": {
                    "host": "db.example.com",
                    "port": 5432,
                    "user": "admin",
                    "password": "pwd",
                    "db_name": "test_db",
                },
            },
            "local_sqlite": {
                "provider": "sqlite",
                "config": {
                    "path": "local.db",
                },
            },
        },
    }

    storage_conf = DatabasesConf.parse(input_dict)

    # Neo4j check
    neo_conf = storage_conf.neo4j_confs["my_neo4j"]
    assert isinstance(neo_conf, Neo4jConf)
    assert neo_conf.host == "localhost"
    assert neo_conf.port == 7687

    # Postgres check
    pg_conf = storage_conf.relational_db_confs["main_postgres"]
    assert isinstance(pg_conf, SqlAlchemyConf)
    assert pg_conf.dialect == "postgresql"
    assert pg_conf.driver == "asyncpg"
    assert pg_conf.host == "db.example.com"
    assert pg_conf.user == "admin"
    assert pg_conf.password == SecretStr("pwd")
    assert pg_conf.db_name == "test_db"
    assert pg_conf.port == 5432
    assert pg_conf.path is None
    assert pg_conf.uri == "postgresql+asyncpg://admin:pwd@db.example.com:5432/test_db"

    # Sqlite check
    sqlite_conf = storage_conf.relational_db_confs["local_sqlite"]
    assert sqlite_conf.dialect == "sqlite"
    assert sqlite_conf.driver == "aiosqlite"
    assert sqlite_conf.path == "local.db"
    assert isinstance(sqlite_conf, SqlAlchemyConf)
    assert sqlite_conf.uri == "sqlite+aiosqlite:///local.db"


def test_parse_unknown_provider_raises():
    input_dict = {
        "databases": {"bad_storage": {"provider": "unknown_db", "host": "localhost"}},
    }
    message = "Supported providers are: neo4j, postgres, sqlite"
    with pytest.raises(ValueError, match=message):
        DatabasesConf.parse(input_dict)


def test_parse_empty_storage_returns_empty_conf():
    input_dict = {"databases": {}}
    storage_conf = DatabasesConf.parse(input_dict)
    assert storage_conf.neo4j_confs == {}
    assert storage_conf.relational_db_confs == {}

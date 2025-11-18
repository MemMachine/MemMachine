import pytest

from memmachine.common.configuration.storage_conf import (
    Neo4JConf,
    SqlAlchemyConf,
    StoragesConf,
)


def test_parse_valid_storage_dict():
    input_dict = {
        "storage": {
            "my_neo4j": {
                "provider": "neo4j",
                "host": "localhost",
                "port": 7687,
                "user": "neo4j",
                "password": "secret",
            },
            "main_postgres": {
                "provider": "postgres",
                "host": "db.example.com",
                "port": 5432,
                "user": "admin",
                "db_name": "test_db",
                "password": "pwd",
            },
            "local_sqlite": {"provider": "sqlite", "host": "local.db"},
        },
    }

    storage_conf = StoragesConf.parse_storage_conf(input_dict)

    # Neo4J check
    neo_conf = storage_conf.neo4j_confs["my_neo4j"]
    assert isinstance(neo_conf, Neo4JConf)
    assert neo_conf.host == "localhost"
    assert neo_conf.port == 7687

    # Postgres check
    pg_conf = storage_conf.relational_db_confs["main_postgres"]
    assert isinstance(pg_conf, SqlAlchemyConf)
    assert pg_conf.db_name == "test_db"
    assert pg_conf.port == 5432

    # Sqlite check
    sqlite_conf = storage_conf.relational_db_confs["local_sqlite"]
    assert isinstance(sqlite_conf, SqlAlchemyConf)
    assert sqlite_conf.host == "local.db"


def test_parse_unknown_provider_raises():
    input_dict = {
        "storage": {"bad_storage": {"provider": "unknown_db", "host": "localhost"}},
    }
    with pytest.raises(ValueError, match="Unknown provider 'unknown_db'"):
        StoragesConf.parse_storage_conf(input_dict)


def test_parse_empty_storage_returns_empty_conf():
    input_dict = {"storage": {}}
    storage_conf = StoragesConf.parse_storage_conf(input_dict)
    assert storage_conf.neo4j_confs == {}
    assert storage_conf.relational_db_confs == {}

import pytest
import yaml
from pydantic import SecretStr

from memmachine.common.configuration.database_conf import (
    DatabasesConf,
    Neo4jConf,
    SqlAlchemyConf,
    SupportedDB,
)


def test_parse_supported_db_enums():
    assert SupportedDB.from_provider("neo4j") == SupportedDB.NEO4J
    assert SupportedDB.from_provider("postgres") == SupportedDB.POSTGRES
    assert SupportedDB.from_provider("sqlite") == SupportedDB.SQLITE

    neo4j_db = SupportedDB.NEO4J
    assert neo4j_db.is_neo4j
    assert neo4j_db.conf_cls == Neo4jConf

    pg_db = SupportedDB.POSTGRES
    assert not pg_db.is_neo4j
    assert pg_db.conf_cls == SqlAlchemyConf
    assert pg_db.dialect == "postgresql"
    assert pg_db.driver == "asyncpg"

    sqlite_db = SupportedDB.SQLITE
    assert not sqlite_db.is_neo4j
    assert sqlite_db.conf_cls == SqlAlchemyConf
    assert sqlite_db.dialect == "sqlite"
    assert sqlite_db.driver == "aiosqlite"


def test_sqlite_without_path_raises():
    message = "non-empty 'path'"
    with pytest.raises(ValueError, match=message):
        SupportedDB.SQLITE.build_config({"uri": "sqlite.db"})


def test_sqlite_with_path_succeeds():
    config = SupportedDB.SQLITE.build_config({"path": "sqlite.db"})
    assert isinstance(config, SqlAlchemyConf)
    assert config.path == "sqlite.db"
    assert config.uri == "sqlite+aiosqlite:///sqlite.db"


def test_invalid_provider_raises():
    message = "Supported providers are"
    with pytest.raises(ValueError, match=message):
        SupportedDB.from_provider("invalid_db")


@pytest.fixture
def db_conf_dict() -> dict:
    return {
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


def test_parse_valid_storage_dict(db_conf_dict):
    storage_conf = DatabasesConf.parse(db_conf_dict)

    # Neo4j check
    neo_conf = storage_conf.neo4j_confs["my_neo4j"]
    assert isinstance(neo_conf, Neo4jConf)
    assert neo_conf.host == "localhost"
    assert neo_conf.port == 7687
    assert neo_conf.user == "neo4j"
    assert neo_conf.password == SecretStr("secret")

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


def test_serialize_deserialize_database_conf(db_conf_dict):
    conf = DatabasesConf.parse(db_conf_dict)
    yaml_str = conf.to_yaml()
    conf_cp = DatabasesConf.parse(yaml.safe_load(yaml_str))
    assert conf == conf_cp


def test_neo4j_uri():
    conf = Neo4jConf(uri="bolt://localhost:1234")
    assert conf.get_uri() == "bolt://localhost:1234"


def test_neo4j_uri_with_host_and_port():
    conf = Neo4jConf(host="neo4j", port=4321)
    assert conf.get_uri() == "bolt://neo4j:4321"


def test_neo4j_uri_with_special_host():
    conf = Neo4jConf(host="neo4j+s://xyz", port=3456)
    assert conf.get_uri() == "neo4j+s://xyz"

import pytest
import yaml
from pydantic import SecretStr

from memmachine_server.common.configuration.database_conf import (
    DatabasesConf,
    Neo4jConf,
    QdrantConf,
    SqlAlchemyConf,
    SupportedDB,
)


def test_parse_supported_db_enums():
    assert SupportedDB.from_provider("neo4j") == SupportedDB.NEO4J
    assert SupportedDB.from_provider("postgres") == SupportedDB.POSTGRES
    assert SupportedDB.from_provider("sqlite") == SupportedDB.SQLITE
    assert SupportedDB.from_provider("qdrant") == SupportedDB.QDRANT

    neo4j_db = SupportedDB.NEO4J
    assert neo4j_db.conf_cls == Neo4jConf

    pg_db = SupportedDB.POSTGRES
    assert pg_db.conf_cls == SqlAlchemyConf
    assert pg_db.dialect == "postgresql"
    assert pg_db.driver == "asyncpg"

    sqlite_db = SupportedDB.SQLITE
    assert sqlite_db.conf_cls == SqlAlchemyConf
    assert sqlite_db.dialect == "sqlite"
    assert sqlite_db.driver == "aiosqlite"

    qdrant_db = SupportedDB.QDRANT
    assert qdrant_db.conf_cls == QdrantConf
    assert qdrant_db.dialect is None
    assert qdrant_db.driver is None


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
            "my_qdrant": {
                "provider": "qdrant",
                "config": {
                    "host": "qdrant.example.com",
                    "port": 6333,
                    "grpc_port": 6334,
                    "prefer_grpc": True,
                    "api_key": "test-key",
                    "is_distributed": True,
                    "registry_replication_factor": 3,
                },
            },
        },
    }


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for var in ["MY_NEO4J_PASSWORD", "MY_DB_PASSWORD"]:
        monkeypatch.delenv(var, raising=False)


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

    # Qdrant check
    qdrant_conf = storage_conf.qdrant_confs["my_qdrant"]
    assert isinstance(qdrant_conf, QdrantConf)
    assert qdrant_conf.host == "qdrant.example.com"
    assert qdrant_conf.port == 6333
    assert qdrant_conf.grpc_port == 6334
    assert qdrant_conf.prefer_grpc is True
    assert qdrant_conf.api_key == SecretStr("test-key")
    assert qdrant_conf.is_distributed is True
    assert qdrant_conf.registry_replication_factor == 3


def test_read_db_password_from_env(monkeypatch, db_conf_dict):
    monkeypatch.setenv("MY_DB_PASSWORD", "env-db-password")
    db_conf_dict["databases"]["main_postgres"]["config"]["password"] = (
        "${MY_DB_PASSWORD}"
    )
    storage_conf = DatabasesConf.parse(db_conf_dict)

    pg_conf = storage_conf.relational_db_confs["main_postgres"]
    assert pg_conf.password == SecretStr("env-db-password")


def test_read_neo4j_password_from_env(monkeypatch, db_conf_dict):
    monkeypatch.setenv("MY_NEO4J_PASSWORD", "env-neo4j-password")
    db_conf_dict["databases"]["my_neo4j"]["config"]["password"] = "${MY_NEO4J_PASSWORD}"
    storage_conf = DatabasesConf.parse(db_conf_dict)

    neo_conf = storage_conf.neo4j_confs["my_neo4j"]
    assert neo_conf.password == SecretStr("env-neo4j-password")


def test_parse_unknown_provider_raises():
    input_dict = {
        "databases": {"bad_storage": {"provider": "unknown_db", "host": "localhost"}},
    }
    message = "Supported providers are: neo4j, postgres, sqlite, nebula_graph, qdrant"
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


def test_neo4j_pool_lifecycle_fields():
    # Defaults are None (use driver's internal defaults)
    conf = Neo4jConf()
    assert conf.max_connection_lifetime is None
    assert conf.liveness_check_timeout is None

    # Explicit values round-trip correctly
    conf = Neo4jConf(max_connection_lifetime=3000.0, liveness_check_timeout=300.0)
    assert conf.max_connection_lifetime == 3000.0
    assert conf.liveness_check_timeout == 300.0


def test_sqlalchemy_pool_lifecycle_fields():
    # Defaults are None (use SQLAlchemy's internal defaults)
    conf = SqlAlchemyConf(dialect="sqlite", driver="aiosqlite", path="test.db")
    assert conf.pool_timeout is None
    assert conf.pool_recycle is None
    assert conf.pool_pre_ping is None

    # Explicit values round-trip correctly
    conf = SqlAlchemyConf(
        dialect="sqlite",
        driver="aiosqlite",
        path="test.db",
        pool_timeout=30,
        pool_recycle=3000,
        pool_pre_ping=True,
    )
    assert conf.pool_timeout == 30
    assert conf.pool_recycle == 3000
    assert conf.pool_pre_ping is True


def test_neo4j_uri():
    conf = Neo4jConf(uri="bolt://localhost:1234")
    assert conf.get_uri() == "bolt://localhost:1234"


def test_neo4j_uri_with_host_and_port():
    conf = Neo4jConf(host="neo4j", port=4321)
    assert conf.get_uri() == "bolt://neo4j:4321"


def test_neo4j_uri_with_special_host():
    conf = Neo4jConf(host="neo4j+s://xyz", port=3456)
    assert conf.get_uri() == "neo4j+s://xyz"


def test_qdrant_conf_defaults():
    conf = QdrantConf()
    assert conf.host == "localhost"
    assert conf.port == 6333
    assert conf.grpc_port == 6334
    assert conf.prefer_grpc is False
    assert conf.https is False
    assert conf.is_distributed is False
    assert conf.registry_replication_factor == 1
    assert conf.api_key.get_secret_value() == ""


def test_qdrant_conf_api_key_from_env(monkeypatch):
    monkeypatch.setenv("QDRANT_API_KEY", "env-qdrant-key")
    conf = QdrantConf(api_key=SecretStr("$QDRANT_API_KEY"))
    assert conf.api_key == SecretStr("env-qdrant-key")


def test_qdrant_build_config():
    config = SupportedDB.QDRANT.build_config(
        {"host": "qdrant.local", "port": 9333, "is_distributed": True}
    )
    assert isinstance(config, QdrantConf)
    assert config.host == "qdrant.local"
    assert config.port == 9333
    assert config.is_distributed is True

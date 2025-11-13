import pytest
from pydantic import SecretStr, ValidationError

from memmachine.common.configuration.storage_conf import (
    DBConf,
    Neo4JConf,
    PostgresConf,
    SqliteConf,
    StorageConf,
    SupportedDB,
)


def test_valid_postgres_config():
    conf = DBConf(
        vendor_name=SupportedDB.POSTGRES,
        host="localhost",
        port=5432,
        user="admin",
        password=SecretStr("secret"),
    )
    assert conf.vendor_name == SupportedDB.POSTGRES
    assert conf.host == "localhost"
    assert conf.port == 5432
    assert conf.user == "admin"
    assert conf.password == SecretStr("secret")


@pytest.mark.parametrize(
    "field,value",
    [
        ("host", ""),
        ("user", ""),
        ("password", ""),
    ],
)
def test_empty_field_raises_for_non_sqlite(field, value):
    kwargs = dict(
        vendor_name=SupportedDB.NEO4J,
        host="localhost",
        port=7687,
        user="neo4j",
        password="secret",
    )
    kwargs[field] = value
    with pytest.raises(ValidationError):
        DBConf(**kwargs)


@pytest.mark.parametrize("port", [0, -1, 70000])
def test_invalid_port_raises(port):
    with pytest.raises(ValidationError):
        DBConf(
            vendor_name=SupportedDB.POSTGRES,
            host="localhost",
            port=port,
            user="admin",
            password=SecretStr("secret"),
        )


@pytest.mark.parametrize(
    "host,port,user,password",
    [
        ("", 0, "", ""),
        ("anything", 12345, "someone", "pwd"),
    ],
)
def test_sqlite_allows_any_values(host, port, user, password):
    conf = DBConf(
        vendor_name=SupportedDB.SQLITE,
        host=host,
        port=port,
        user=user,
        password=SecretStr(password),
    )
    assert conf.vendor_name == SupportedDB.SQLITE
    assert conf.host == host
    assert conf.port == port
    assert conf.user == user
    assert conf.password == SecretStr(password)


def test_parse_valid_storage_dict():
    input_dict = {
        "storage": {
            "my_neo4j": {
                "vendor_name": "neo4j",
                "host": "localhost",
                "port": 7687,
                "user": "neo4j",
                "password": "secret",
            },
            "main_postgres": {
                "vendor_name": "postgres",
                "host": "db.example.com",
                "port": 5432,
                "user": "admin",
                "db_name": "test_db",
                "password": "pwd",
            },
            "local_sqlite": {"vendor_name": "sqlite", "file_path": "local.db"},
        }
    }

    storage_conf = StorageConf.parse_storage_conf(input_dict)

    # Neo4J check
    neo_conf = storage_conf.neo4j_confs["my_neo4j"]
    assert isinstance(neo_conf, Neo4JConf)
    assert neo_conf.host == "localhost"
    assert neo_conf.port == 7687

    # Postgres check
    pg_conf = storage_conf.postgres_confs["main_postgres"]
    assert isinstance(pg_conf, PostgresConf)
    assert pg_conf.db_name == "test_db"
    assert pg_conf.port == 5432

    # Sqlite check
    sqlite_conf = storage_conf.sqlite_confs["local_sqlite"]
    assert isinstance(sqlite_conf, SqliteConf)
    assert sqlite_conf.file_path == "local.db"


def test_parse_unknown_vendor_raises():
    input_dict = {
        "storage": {"bad_storage": {"vendor_name": "unknown_db", "host": "localhost"}}
    }
    with pytest.raises(ValueError, match="Unknown vendor_name 'unknown_db'"):
        StorageConf.parse_storage_conf(input_dict)


def test_parse_empty_storage_returns_empty_conf():
    input_dict = {"storage": {}}
    storage_conf = StorageConf.parse_storage_conf(input_dict)
    assert storage_conf.neo4j_confs == {}
    assert storage_conf.postgres_confs == {}
    assert storage_conf.sqlite_confs == {}

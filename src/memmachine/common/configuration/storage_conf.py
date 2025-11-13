from enum import Enum
from typing import Self

from pydantic import BaseModel, Field, SecretStr, model_validator


class Neo4JConf(BaseModel):
    host: str = Field(default="localhost", description="neo4j connection host")
    port: int = Field(default=7687, description="neo4j connection port")
    user: str = Field(default="neo4j", description="neo4j username")
    password: SecretStr = Field(
        default=SecretStr("neo4j_password"), description="neo4j database password"
    )
    max_concurrent_transactions: int = Field(
        default=100, description="Maximum number of concurrent transactions", gt=0
    )
    force_exact_similarity_search: bool = Field(
        default=False, description="Whether to force exact similarity search"
    )


class PostgresConf(BaseModel):
    host: str = Field(default="localhost", description="PostgreSQL connection host")
    port: int = Field(default=5432, description="PostgreSQL connection port")
    user: str = Field(default="memmachine", description="PostgreSQL username")
    password: SecretStr = Field(
        default=SecretStr("memmachine_password"),
        description="PostgreSQL database password",
    )
    db_name: str = Field(default="memmachine", description="PostgreSQL database name")
    vector_schema: str = Field(
        default="public", description="PostgreSQL schema for vector data"
    )
    statement_cache_size: int = Field(
        default=0, description="PostgreSQL statement cache size (0 to disable)"
    )


class SqliteConf(BaseModel):
    file_path: str = Field(
        default="memmachine.db", description="SQLite database file path"
    )


class SupportedDB(str, Enum):
    NEO4J = "neo4j"
    POSTGRES = "postgres"
    SQLITE = "sqlite"


class DBConf(BaseModel):
    vendor_name: SupportedDB = Field(..., description="Database vendor type")
    host: str = Field(default="", description="Database host (ignored for SQLite)")
    port: int = Field(default=0, description="Database port (ignored for SQLite)")
    user: str = Field(default="", description="Database username (ignored for SQLite)")
    password: SecretStr = Field(
        default=SecretStr(""), description="Database password (ignored for SQLite)"
    )

    @model_validator(mode="after")
    @staticmethod
    def validate_by_vendor(values):
        if values.vendor_name == SupportedDB.SQLITE:
            # sqlite does not need host/port/user/password
            return values

        # For other vendors, all fields must be valid/non-empty
        if not values.host:
            raise ValueError("host must not be empty for non-sqlite databases")
        if not (1 <= values.port <= 65535):
            raise ValueError(
                "port must be between 1 and 65535 for non-sqlite databases"
            )
        if not values.user:
            raise ValueError("user must not be empty for non-sqlite databases")
        if not values.password:
            raise ValueError("password must not be empty for non-sqlite databases")

        return values


class StorageConf(BaseModel):
    neo4j_confs: dict[str, Neo4JConf] = {}
    postgres_confs: dict[str, PostgresConf] = {}
    sqlite_confs: dict[str, SqliteConf] = {}

    @classmethod
    def parse_storage_conf(cls, input_dict: dict) -> Self:
        storage = input_dict
        for key in ["storage", "Storage"]:
            if key in input_dict:
                storage = input_dict.get(key, {})

        neo4j_dict, pg_dict, sqlite_dict = {}, {}, {}

        for storage_id, conf in storage.items():
            vendor = conf.get("vendor_name").lower()
            if vendor == "neo4j":
                neo4j_dict[storage_id] = Neo4JConf(**conf)
            elif vendor == "postgres":
                pg_dict[storage_id] = PostgresConf(**conf)
            elif vendor == "sqlite":
                sqlite_dict[storage_id] = SqliteConf(**conf)
            else:
                raise ValueError(
                    f"Unknown vendor_name '{vendor}' for storage_id '{storage_id}'"
                )

        return cls(
            neo4j_confs=neo4j_dict,
            postgres_confs=pg_dict,
            sqlite_confs=sqlite_dict
        )

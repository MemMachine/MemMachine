"""Storage configuration models."""

from enum import Enum
from typing import ClassVar, Self

import yaml
from pydantic import BaseModel, Field, SecretStr, model_validator

from memmachine.common.configuration.mixin_confs import (
    PasswordMixin,
    YamlSerializableMixin,
)


class Neo4jConf(YamlSerializableMixin, PasswordMixin):
    """Configuration options for a Neo4j instance."""

    uri: str = Field(default="", description="Neo4j database URI")
    host: str = Field(default="localhost", description="neo4j connection host")
    port: int = Field(default=7687, description="neo4j connection port")
    user: str = Field(default="neo4j", description="neo4j username")
    password: SecretStr = Field(
        default=SecretStr("neo4j_password"),
        description=(
            "Password for the Neo4j database user. "
            "If not explicitly set, a default placeholder value is used. "
            "You may reference an environment variable using `$ENV` or `${ENV}` "
            "syntax (for example, `$NEO4J_PASSWORD`)."
        ),
    )
    force_exact_similarity_search: bool = Field(
        default=False,
        description="Whether to force exact similarity search",
    )
    range_index_creation_threshold: int | None = Field(
        default=None,
        description=(
            "Minimum number of entities in a collection or relationship "
            "required before Neo4j automatically creates a range index."
        ),
    )
    vector_index_creation_threshold: int | None = Field(
        default=None,
        description=(
            "Minimum number of entities in a collection or relationship "
            "required before Neo4j automatically creates a vector index."
        ),
    )

    def get_uri(self) -> str:
        if self.uri:
            return self.uri
        if "neo4j+s://" in self.host:
            return self.host
        return f"bolt://{self.host}:{self.port}"


class SqlAlchemyConf(YamlSerializableMixin, PasswordMixin):
    """Configuration for SQLAlchemy-backed relational databases."""

    dialect: str = Field(..., description="SQL dialect")
    driver: str = Field(..., description="SQLAlchemy driver")

    host: str | None = Field(default=None, description="DB connection host")
    path: str | None = Field(default=None, description="DB file path")
    port: int | None = Field(default=None, description="DB connection port")
    user: str | None = Field(default=None, description="DB username")
    password: SecretStr | None = Field(
        default=None,
        description=(
            "Optional password for the database user. "
            "You can reference an environment variable using `$ENV` or `${ENV}` syntax "
            "(for example, `$DB_PASSWORD`)."
        ),
    )
    db_name: str | None = Field(default=None, description="DB name")
    pool_size: int | None = Field(
        default=None,
        description=(
            "Number of persistent connections to maintain in the connection pool. "
            "If set, the pool will keep up to this many open connections ready for use."
        ),
    )
    max_overflow: int | None = Field(
        default=None,
        description=(
            "Maximum number of temporary connections allowed above `pool_size` during "
            "traffic spikes. These overflow connections are created on demand and "
            "disposed of when no longer needed."
        ),
    )

    @property
    def schema_part(self) -> str:
        """Construct the SQLAlchemy database schema."""
        return f"{self.dialect}+{self.driver}://"

    @property
    def auth_part(self) -> str:
        """Construct the SQLAlchemy database credentials part."""
        auth_part = ""
        if self.user and self.password:
            auth_part = f"{self.user}:{self.password.get_secret_value()}@"
        elif self.user:
            auth_part = f"{self.user}@"
        return auth_part

    @property
    def host_and_port(self) -> str:
        """Construct the host and port part of the URI."""
        host_part = self.host or ""
        if self.port:
            host_part += f":{self.port}"
        return host_part

    @property
    def path_or_db(self) -> str:
        """Construct the path part of the URI."""
        ret = f"/{self.path}" if self.path else ""
        ret += f"/{self.db_name}" if self.db_name else ""
        return ret

    @property
    def uri(self) -> str:
        """Construct the SQLAlchemy database URI."""
        return (
            f"{self.schema_part}{self.auth_part}{self.host_and_port}{self.path_or_db}"
        )

    @model_validator(mode="after")
    def validate_sqlite(self) -> Self:
        if self.dialect == "sqlite" and not self.path:
            raise ValueError("SQLite requires a non-empty 'path'")
        return self


class SupportedDB(str, Enum):
    """Supported database providers."""

    # <-- Add these annotations so mypy knows these attributes exist
    conf_cls: type[Neo4jConf] | type[SqlAlchemyConf]
    dialect: str | None
    driver: str | None

    NEO4J = ("neo4j", Neo4jConf, None, None)
    POSTGRES = ("postgres", SqlAlchemyConf, "postgresql", "asyncpg")
    SQLITE = ("sqlite", SqlAlchemyConf, "sqlite", "aiosqlite")

    def __new__(
        cls,
        value: str,
        conf_cls: type[Neo4jConf] | type[SqlAlchemyConf],
        dialect: str | None,
        driver: str | None,
    ) -> Self:
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.conf_cls = conf_cls  # mypy now knows these attributes exist
        obj.dialect = dialect
        obj.driver = driver
        return obj

    @classmethod
    def from_provider(cls, provider: str) -> Self:
        for m in cls:
            if m.value == provider:
                return m
        valid = ", ".join(str(m.value) for m in cls)
        raise ValueError(
            f"Unsupported provider '{provider}'. Supported providers are: {valid}"
        )

    def build_config(self, conf: dict) -> Neo4jConf | SqlAlchemyConf:
        if self is SupportedDB.NEO4J:
            return self.conf_cls(**conf)
        conf_copy = {**conf, "dialect": self.dialect, "driver": self.driver}
        return self.conf_cls(**conf_copy)

    @property
    def is_neo4j(self) -> bool:
        return self is SupportedDB.NEO4J


class DatabasesConf(BaseModel):
    """Top-level storage configuration mapping identifiers to backends."""

    neo4j_confs: dict[str, Neo4jConf] = {}
    relational_db_confs: dict[str, SqlAlchemyConf] = {}

    PROVIDER_KEY: ClassVar[str] = "provider"
    CONFIG_KEY: ClassVar[str] = "config"
    NEO4J: ClassVar[str] = "neo4j"
    RELATIONAL_DB: ClassVar[str] = "relational-db"
    POSTGRES: ClassVar[str] = "postgres"
    POSTGRESQL: ClassVar[str] = "postgresql"
    SQLITE: ClassVar[str] = "sqlite"
    DIALECT: ClassVar[str] = "dialect"

    def to_yaml_dict(self) -> dict:
        """Serialize the database configuration to a YAML-compatible dictionary."""
        databases: dict[str, dict] = {}

        def add_database(db_id: str, db_type: str, config: dict) -> None:
            provider = self.SQLITE
            if db_type == self.NEO4J:
                provider = self.NEO4J
            elif db_type == self.RELATIONAL_DB:
                dialect = config.get(self.DIALECT)
                if dialect == self.POSTGRESQL:
                    provider = self.POSTGRES
                elif dialect == self.SQLITE:
                    provider = self.SQLITE
            databases[db_id] = {
                self.PROVIDER_KEY: provider,
                self.CONFIG_KEY: config,
            }

        for database_id, conf in self.neo4j_confs.items():
            add_database(database_id, self.NEO4J, conf.to_yaml_dict())

        for database_id, conf in self.relational_db_confs.items():
            add_database(database_id, self.RELATIONAL_DB, conf.to_yaml_dict())

        return databases

    def to_yaml(self) -> str:
        data = {"databases": self.to_yaml_dict()}
        return yaml.safe_dump(data, sort_keys=True)

    @classmethod
    def parse(cls, input_dict: dict) -> Self:
        databases = input_dict.get("databases", {})

        if isinstance(databases, cls):
            return databases

        neo4j_dict = {}
        relational_db_dict = {}

        for database_id, resource_definition in databases.items():
            provider_str = resource_definition.get(cls.PROVIDER_KEY)
            conf = resource_definition.get(cls.CONFIG_KEY, {})

            provider = SupportedDB.from_provider(provider_str)
            config_obj = provider.build_config(conf)

            if provider.is_neo4j:
                neo4j_dict[database_id] = config_obj
            else:
                relational_db_dict[database_id] = config_obj

        return cls(
            neo4j_confs=neo4j_dict,
            relational_db_confs=relational_db_dict,
        )

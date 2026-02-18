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
    max_connection_pool_size: int | None = Field(
        default=None,
        description=(
            "Maximum number of connections to maintain in the connection pool. "
            "Internal default is 100."
        ),
    )
    connection_acquisition_timeout: float | None = Field(
        default=None,
        description=(
            "Maximum time in seconds to wait for a connection from the pool. "
            "Internal default is 60.0."
        ),
    )

    def get_uri(self) -> str:
        if self.uri:
            return self.uri
        if "neo4j+s://" in self.host:
            return self.host
        return f"bolt://{self.host}:{self.port}"


class NebulaGraphConf(YamlSerializableMixin, PasswordMixin):
    """Configuration options for a NebulaGraph Enterprise instance."""

    hosts: list[str] = Field(
        default_factory=lambda: ["127.0.0.1:9669"],
        description="List of NebulaGraph graphd service addresses (host:port format)",
    )
    username: str = Field(default="root", description="NebulaGraph username")
    password: SecretStr = Field(
        default=SecretStr("nebula"),
        description=(
            "Password for the NebulaGraph database user. "
            "You may reference an environment variable using `$ENV` or `${ENV}` "
            "syntax (for example, `$NEBULA_PASSWORD`)."
        ),
    )

    # Schema and Graph (Enterprise model)
    schema_name: str = Field(
        default="/default_schema",
        description=(
            "NebulaGraph schema path. Default is '/default_schema'. "
            "A schema is a logical container for graph types and graphs."
        ),
    )
    graph_type_name: str = Field(
        default="memmachine_type",
        description=(
            "Graph type name. This defines the schema (node types, edge types, "
            "and their properties). Multiple graphs can share the same graph type. "
            "Will be created if it doesn't exist."
        ),
    )
    graph_name: str = Field(
        default="memmachine",
        description=(
            "Graph name within the schema. This is the actual graph instance "
            "that will be used for storing data. Will be created if it doesn't exist."
        ),
    )

    # Session pooling configuration
    session_pool_size: int = Field(
        default=4,
        description=(
            "Number of sessions in the session pool. "
            "More sessions allow higher concurrency but use more resources."
        ),
    )
    session_pool_wait_timeout: float = Field(
        default=60.0,
        description=(
            "Maximum time (in seconds) to wait for a session from the pool. "
            "If None or 0, will wait indefinitely."
        ),
    )

    # Search behavior
    force_exact_similarity_search: bool = Field(
        default=False,
        description="Whether to force exact similarity search instead of ANN",
    )

    # Index creation thresholds
    range_index_creation_threshold: int | None = Field(
        default=None,
        description=(
            "Minimum number of entities in a collection or relationship "
            "required before NebulaGraph automatically creates a normal index."
        ),
    )
    vector_index_creation_threshold: int | None = Field(
        default=None,
        description=(
            "Minimum number of entities in a collection or relationship "
            "required before NebulaGraph automatically creates a vector index. "
            "Vector indexes enable ANN (Approximate Nearest Neighbor) search for fast "
            "similarity queries on large datasets. Below this threshold, KNN (K-Nearest "
            "Neighbor) exact search is used, which does not require an index and is "
            "suitable for small-sized graphs and low-dimensional vectors."
        ),
    )

    # Vector index tuning parameters
    ann_index_type: str = Field(
        default="IVF",
        description=(
            "Vector index type: 'IVF' for balanced performance (default), "
            "'HNSW' for higher recall. IVF is faster to build and more memory-efficient, "
            "while HNSW provides better recall at the cost of slower indexing."
        ),
    )
    ivf_nlist: int = Field(
        default=256,
        description=(
            "IVF index parameter: Number of clusters (NLIST). "
            "Higher values = more accurate search, slower index build. "
            "Recommended: 256 for balanced, 1024 for large datasets."
        ),
    )
    ivf_nprobe: int = Field(
        default=8,
        description=(
            "IVF query parameter: Number of clusters to search (NPROBE). "
            "Higher values = better recall, slower queries. "
            "Recommended: 4 (fast, ~70% recall), 8 (balanced, ~85%), 16+ (slow, ~95%)."
        ),
    )
    hnsw_max_degree: int = Field(
        default=16,
        description=(
            "HNSW index parameter: Maximum neighbors per node (MAXDEGREE). "
            "Higher values = better recall, more memory usage. "
            "Recommended: 16 (default), 32 (high precision)."
        ),
    )
    hnsw_ef_construction: int = Field(
        default=200,
        description=(
            "HNSW index parameter: Construction quality (EFCONSTRUCTION). "
            "Higher values = better index quality, slower build. "
            "Recommended: 200 (default), 400+ (high quality)."
        ),
    )
    hnsw_ef_search: int = Field(
        default=40,
        description=(
            "HNSW query parameter: Search quality (EFSEARCH). "
            "Higher values = better recall, slower queries. "
            "Recommended: 20 (fast, ~80% recall), 40 (balanced, ~90%), 100+ (slow, ~98%)."
        ),
    )

    def get_hosts(self) -> list[str]:
        """Get the list of NebulaGraph host addresses."""
        return self.hosts


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
    conf_cls: type[Neo4jConf] | type[SqlAlchemyConf] | type[NebulaGraphConf]
    dialect: str | None
    driver: str | None

    NEO4J = ("neo4j", Neo4jConf, None, None)
    POSTGRES = ("postgres", SqlAlchemyConf, "postgresql", "asyncpg")
    SQLITE = ("sqlite", SqlAlchemyConf, "sqlite", "aiosqlite")
    NEBULA_GRAPH = ("nebula_graph", NebulaGraphConf, None, None)

    def __new__(
        cls,
        value: str,
        conf_cls: type[Neo4jConf] | type[SqlAlchemyConf] | type[NebulaGraphConf],
        dialect: str | None,
        driver: str | None,
    ) -> Self:
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.conf_cls = conf_cls  # type checker now knows these attributes exist
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

    def build_config(self, conf: dict) -> Neo4jConf | SqlAlchemyConf | NebulaGraphConf:
        if self is SupportedDB.NEO4J:
            return self.conf_cls(**conf)
        if self is SupportedDB.NEBULA_GRAPH:
            return self.conf_cls(**conf)
        # Relational DBs (PostgreSQL, SQLite)
        conf_copy = {**conf, "dialect": self.dialect, "driver": self.driver}
        return self.conf_cls(**conf_copy)

    @property
    def is_neo4j(self) -> bool:
        return self is SupportedDB.NEO4J

    @property
    def is_nebula_graph(self) -> bool:
        return self is SupportedDB.NEBULA_GRAPH


class DatabasesConf(BaseModel):
    """Top-level storage configuration mapping identifiers to backends."""

    neo4j_confs: dict[str, Neo4jConf] = {}
    relational_db_confs: dict[str, SqlAlchemyConf] = {}
    nebula_graph_confs: dict[str, NebulaGraphConf] = {}

    PROVIDER_KEY: ClassVar[str] = "provider"
    CONFIG_KEY: ClassVar[str] = "config"
    NEO4J: ClassVar[str] = "neo4j"
    RELATIONAL_DB: ClassVar[str] = "relational-db"
    POSTGRES: ClassVar[str] = "postgres"
    POSTGRESQL: ClassVar[str] = "postgresql"
    SQLITE: ClassVar[str] = "sqlite"
    NEBULA_GRAPH: ClassVar[str] = "nebula_graph"
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

        for database_id, conf in self.nebula_graph_confs.items():
            databases[database_id] = {
                self.PROVIDER_KEY: self.NEBULA_GRAPH,
                self.CONFIG_KEY: conf.to_yaml_dict(),
            }

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
        nebula_graph_dict = {}

        for database_id, resource_definition in databases.items():
            provider_str = resource_definition.get(cls.PROVIDER_KEY)
            conf = resource_definition.get(cls.CONFIG_KEY, {})

            provider = SupportedDB.from_provider(provider_str)
            config_obj = provider.build_config(conf)

            if provider.is_neo4j:
                neo4j_dict[database_id] = config_obj
            elif provider.is_nebula_graph:
                nebula_graph_dict[database_id] = config_obj
            else:
                relational_db_dict[database_id] = config_obj

        return cls(
            neo4j_confs=neo4j_dict,
            relational_db_confs=relational_db_dict,
            nebula_graph_confs=nebula_graph_dict,
        )

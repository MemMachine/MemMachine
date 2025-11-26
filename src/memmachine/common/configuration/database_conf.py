"""Storage configuration models."""

from enum import Enum
from typing import Self

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator


class Neo4jConf(BaseModel):
    """Configuration options for a Neo4j instance."""

    uri: str = Field(default="", description="Neo4j database URI")
    host: str = Field(default="localhost", description="neo4j connection host")
    port: int = Field(default=7687, description="neo4j connection port")
    user: str = Field(default="neo4j", description="neo4j username")
    password: SecretStr = Field(
        default=SecretStr("neo4j_password"),
        description="neo4j database password",
    )
    force_exact_similarity_search: bool = Field(
        default=False,
        description="Whether to force exact similarity search",
    )

    @field_validator("password", mode="before")
    @classmethod
    def convert_password(cls, v: str | SecretStr) -> SecretStr:
        if isinstance(v, SecretStr):
            return v
        if isinstance(v, str):
            return SecretStr(v)
        raise TypeError("password must be a string or SecretStr")

    def get_uri(self) -> str:
        if self.uri:
            return self.uri
        if "neo4j+s://" in self.host:
            return self.host
        return f"bolt://{self.host}:{self.port}"


class SqlAlchemyConf(BaseModel):
    """Configuration for SQLAlchemy-backed relational databases."""

    dialect: str = Field(..., description="SQL dialect")
    driver: str = Field(..., description="SQLAlchemy driver")

    host: str | None = Field(default=None, description="DB connection host")
    path: str | None = Field(default=None, description="DB file path")
    port: int | None = Field(default=None, description="DB connection port")
    user: str | None = Field(default=None, description="DB username")
    password: SecretStr | None = Field(
        default=None,
        description="DB password",
    )
    db_name: str | None = Field(default=None, description="DB name")

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
        return self.conf_cls(
            dialect=self.dialect,
            driver=self.driver,
            **conf,
        )

    @property
    def is_neo4j(self) -> bool:
        return self is SupportedDB.NEO4J


class DatabasesConf(BaseModel):
    """Top-level storage configuration mapping identifiers to backends."""

    neo4j_confs: dict[str, Neo4jConf] = {}
    relational_db_confs: dict[str, SqlAlchemyConf] = {}

    @classmethod
    def parse(cls, input_dict: dict) -> Self:
        databases = input_dict.get("databases", {})

        if isinstance(databases, cls):
            return databases

        neo4j_dict = {}
        relational_db_dict = {}

        for database_id, resource_definition in databases.items():
            provider_str = resource_definition.get("provider")
            conf = resource_definition.get("config", {})

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

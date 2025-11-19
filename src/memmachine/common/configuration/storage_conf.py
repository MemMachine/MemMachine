"""Storage configuration models."""

from enum import Enum
from typing import Self

from pydantic import BaseModel, Field, SecretStr


class Neo4JConf(BaseModel):
    """Configuration options for a Neo4j instance."""

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


class SqlAlchemyConf(BaseModel):
    """Configuration for SQLAlchemy-backed relational databases."""

    dialect: str = Field(..., description="SQL dialect")
    driver: str = Field(..., description="SQLAlchemy driver")

    host: str = Field(..., description="DB connection host")
    port: int | None = Field(default=None, description="DB connection port")
    user: str | None = Field(default=None, description="DB username")
    password: SecretStr | None = Field(
        default=None,
        description="DB password",
    )
    db_name: str | None = Field(default=None, description="DB name")


class SupportedDB(str, Enum):
    NEO4J = ("neo4j", Neo4JConf, None, None)
    POSTGRES = ("postgres", SqlAlchemyConf, "postgresql", "asyncpg")
    SQLITE = ("sqlite", SqlAlchemyConf, "sqlite", "aiosqlite")

    def __new__(cls, value, conf_cls, dialect, driver):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.conf_cls = conf_cls
        obj.dialect = dialect
        obj.driver = driver
        return obj

    @classmethod
    def from_provider(cls, provider: str) -> Self:
        """Convert provider string to enum with a clear error message."""
        try:
            return cls(provider)
        except ValueError:
            valid = ", ".join(str(e.value) for e in cls)
            raise ValueError(
                f"Unsupported provider '{provider}'. Supported providers are: {valid}"
            )

    def build_config(self, conf: dict):
        """Factory method for building the provider-specific config object."""
        if self is self.NEO4J:
            # Neo4J has its own config model
            return self.conf_cls(**conf)

        # All relational DB providers share SqlAlchemyConf
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

    neo4j_confs: dict[str, Neo4JConf] = {}
    relational_db_confs: dict[str, SqlAlchemyConf] = {}

    @classmethod
    def parse(cls, input_dict: dict) -> Self:
        databases = input_dict.get("databases", {})

        neo4j_dict = {}
        relational_db_dict = {}

        for storage_id, resource_definition in databases.items():
            provider_str = resource_definition.get("provider")
            conf = resource_definition.get("config", {})

            provider = SupportedDB.from_provider(provider_str)
            config_obj = provider.build_config(conf)

            if provider.is_neo4j:
                neo4j_dict[storage_id] = config_obj
            else:
                relational_db_dict[storage_id] = config_obj

        return cls(
            neo4j_confs=neo4j_dict,
            relational_db_confs=relational_db_dict,
        )

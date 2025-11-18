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
    """Supported database provider identifiers."""

    NEO4J = "neo4j"
    POSTGRES = "postgres"
    SQLITE = "sqlite"


class StoragesConf(BaseModel):
    """Top-level storage configuration mapping identifiers to backends."""

    neo4j_confs: dict[str, Neo4JConf] = {}
    relational_db_confs: dict[str, SqlAlchemyConf] = {}

    @classmethod
    def parse_storage_conf(cls, input_dict: dict) -> Self:
        """Parse storage configuration from a raw mapping."""
        storage = input_dict
        for key in ["storage", "Storage"]:
            if key in input_dict:
                storage = input_dict.get(key, {})

        neo4j_dict, relational_db_dict = {}, {}

        for storage_id, conf in storage.items():
            provider = conf.get("provider")
            if provider == "neo4j":
                neo4j_dict[storage_id] = Neo4JConf(**conf)
            elif provider == "postgres":
                relational_db_dict[storage_id] = SqlAlchemyConf(
                    dialect="postgresql",
                    driver="asyncpg",
                    **conf,
                )
            elif provider == "sqlite":
                relational_db_dict[storage_id] = SqlAlchemyConf(
                    dialect="sqlite",
                    driver="aiosqlite",
                    **conf,
                )
            else:
                raise ValueError(
                    f"Unknown provider '{provider}' for storage_id '{storage_id}'",
                )

        return cls(
            neo4j_confs=neo4j_dict,
            relational_db_confs=relational_db_dict,
        )

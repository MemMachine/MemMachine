import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from memmachine_server.semantic_memory.storage.sqlalchemy_pgvector_semantic import (
    SqlAlchemyPgVectorSemanticStorage,
)


def test_an_engine_of_another_dialect_is_refused(tmp_path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    with pytest.raises(ValueError, match="sqlite"):
        SqlAlchemyPgVectorSemanticStorage(engine)


def test_an_engine_sharing_one_connection_is_refused() -> None:
    engine = create_async_engine(
        "postgresql+asyncpg://localhost/memmachine", poolclass=StaticPool
    )
    with pytest.raises(ValueError, match="StaticPool"):
        SqlAlchemyPgVectorSemanticStorage(engine)

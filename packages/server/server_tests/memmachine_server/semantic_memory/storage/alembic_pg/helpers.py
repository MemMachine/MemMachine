from __future__ import annotations

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncEngine

SCHEMA_UPGRADER_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = SCHEMA_UPGRADER_ROOT / "src"


async def apply_sql_file(engine: AsyncEngine, sql_path: Path) -> None:
    statements = _read_sql_statements(sql_path)
    if not statements:
        return

    async with engine.begin() as conn:
        for statement in statements:
            await conn.exec_driver_sql(statement)


def _read_sql_statements(sql_path: Path) -> list[str]:
    content = sql_path.read_text()
    statements: list[str] = []
    buffer: list[str] = []

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        buffer.append(line)
        if stripped.endswith(";"):
            statement = "\n".join(buffer).strip().rstrip(";")
            if statement:
                statements.append(statement)
            buffer.clear()

    trailing = "\n".join(buffer).strip().rstrip(";")
    if trailing:
        statements.append(trailing)

    return statements

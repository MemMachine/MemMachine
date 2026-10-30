"""Schema changes in the server's source happen only where they were placed on purpose.

A store creates its durable resources in `migrate`, run by the setup step,
and never in `startup`. The other sites create or drop what the request
path needs per partition, per session, or per semantic set. A new site
fails this test until it is placed in one of the groups below.
"""

import ast
import re
from pathlib import Path

import memmachine_server

_SOURCE = Path(memmachine_server.__file__).parent

_DDL_TEXT = re.compile(
    r"^\s*(CREATE|ALTER|DROP)\s+(TABLE|INDEX|TYPE|EXTENSION|SCHEMA|GRAPH|"
    r"CONSTRAINT|VIRTUAL|VECTOR|RANGE|NORMAL|UNIQUE|OR)\b"
)
_DDL_ATTRIBUTES = {
    "create_all",
    "drop_all",
    "create_collection",
    "create_payload_index",
    "create_index",
    "add_collection_field",
}

_MIGRATE = {
    (
        "common/vector_store/qdrant_vector_store.py",
        "QdrantVectorStore._prepare_storage",
    ),
    (
        "common/vector_store/milvus_vector_store.py",
        "MilvusVectorStore._prepare_storage",
    ),
    (
        "common/vector_store/milvus_vector_store.py",
        "MilvusVectorStore._prepare_storage._create_collection",
    ),
    (
        "common/vector_graph_store/nebula_graph_vector_graph_store.py",
        "NebulaGraphVectorGraphStore.migrate",
    ),
    (
        "semantic_memory/storage/sqlalchemy_pgvector_semantic.py",
        "apply_alembic_migrations",
    ),
    (
        "semantic_memory/storage/sqlalchemy_pgvector_semantic.py",
        "apply_alembic_migrations.run_migrations",
    ),
    (
        "semantic_memory/storage/vector_store_semantic_storage.py",
        "VectorStoreSemanticStorage.migrate",
    ),
    (
        "semantic_memory/storage/neo4j_semantic_storage.py",
        "Neo4jSemanticStorage.migrate",
    ),
    (
        "semantic_memory/config_store/config_store_sqlalchemy.py",
        "SemanticConfigStorageSqlAlchemy.migrate",
    ),
    (
        "semantic_memory/cluster_store/cluster_store_sqlalchemy.py",
        "ClusterStateStorageSqlAlchemy.migrate",
    ),
}

_REQUEST_PATH = {
    # A SQLite vector store's tables of one partition.
    (
        "common/vector_store/sqlite_vector_store.py",
        "SQLiteVectorStore._insert_partition",
    ),
    (
        "common/vector_store/sqlite_vector_store.py",
        "SQLiteVectorStore.delete_partition",
    ),
    (
        "common/vector_store/sqlite_vec_vector_store.py",
        "SQLiteVecVectorStore._insert_partition",
    ),
    (
        "common/vector_store/sqlite_vec_vector_store.py",
        "SQLiteVecVectorStore.delete_partition",
    ),
    # Graph stores' indexes and types of one session.
    (
        "common/vector_graph_store/neo4j_vector_graph_store.py",
        "Neo4jVectorGraphStore._create_range_index_if_not_exists",
    ),
    (
        "common/vector_graph_store/neo4j_vector_graph_store.py",
        "Neo4jVectorGraphStore._create_vector_index_if_not_exists",
    ),
    (
        "common/vector_graph_store/nebula_graph_vector_graph_store.py",
        "NebulaGraphVectorGraphStore._create_range_index_if_not_exists",
    ),
    (
        "common/vector_graph_store/nebula_graph_vector_graph_store.py",
        "NebulaGraphVectorGraphStore._create_vector_index_if_not_exists",
    ),
    (
        "common/vector_graph_store/nebula_graph_vector_graph_store.py",
        "NebulaGraphVectorGraphStore._ensure_graph_type_for_edges",
    ),
    (
        "common/vector_graph_store/nebula_graph_vector_graph_store.py",
        "NebulaGraphVectorGraphStore._ensure_graph_type_for_nodes",
    ),
    (
        "common/vector_graph_store/nebula_graph_vector_graph_store.py",
        "NebulaGraphVectorGraphStore.delete_all_data",
    ),
    # Neo4j semantic memory's index of one set.
    (
        "semantic_memory/storage/neo4j_semantic_storage.py",
        "Neo4jSemanticStorage._create_vector_index",
    ),
    (
        "semantic_memory/storage/neo4j_semantic_storage.py",
        "Neo4jSemanticStorage._drop_index_if_named",
    ),
    (
        "semantic_memory/storage/neo4j_semantic_storage.py",
        "Neo4jSemanticStorage.delete_all",
    ),
}

_TEST_HELPERS = {
    (
        "common/session_manager/session_data_manager_sql_impl.py",
        "SessionDataManagerSQL.drop_tables",
    ),
}


def _is_revision_or_runner(relative: str) -> bool:
    return (
        "/migrations/" in relative
        or "/alembic_pg/" in relative
        or relative.startswith("common/schema_migration/")
    )


class _DDLSites(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scope: list[str] = []
        self.sites: set[str] = set()

    def _visit_scope(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scope(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scope(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        is_alembic_upgrade = (
            node.attr == "upgrade"
            and isinstance(node.value, ast.Name)
            and node.value.id == "command"
        )
        if node.attr in _DDL_ATTRIBUTES or is_alembic_upgrade:
            self.sites.add(".".join(self.scope))
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and _DDL_TEXT.match(node.value):
            self.sites.add(".".join(self.scope))


def test_every_ddl_site_is_placed():
    found = set()
    for path in _SOURCE.rglob("*.py"):
        relative = path.relative_to(_SOURCE).as_posix()
        if _is_revision_or_runner(relative):
            continue
        visitor = _DDLSites()
        visitor.visit(ast.parse(path.read_text()))
        found |= {(relative, qualname) for qualname in visitor.sites}

    assert found - (_MIGRATE | _REQUEST_PATH | _TEST_HELPERS) == set()

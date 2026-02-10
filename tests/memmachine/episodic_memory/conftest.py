"""Shared test fixtures for episodic memory tests."""

import os

import pytest
import pytest_asyncio


@pytest.fixture(scope="module")
def nebula_connection_info_factory():
    """
    Factory fixture for creating NebulaGraph connection info.

    Returns a function that accepts schema_name and graph_name to create
    unique connection info for each test module.

    Environment variables:
    - NEBULA_HOST (default: 127.0.0.1:9669)
    - NEBULA_USER (default: root)
    - NEBULA_PASSWORD (default: nebula)
    """

    def _create_connection_info(schema_name: str, graph_name: str) -> dict:
        nebula_host = os.environ.get("NEBULA_HOST", "127.0.0.1:9669")
        nebula_user = os.environ.get("NEBULA_USER", "root")
        nebula_password = os.environ.get("NEBULA_PASSWORD", "nebula")

        return {
            "hosts": [nebula_host],
            "username": nebula_user,
            "password": nebula_password,
            "schema_name": schema_name,
            "graph_type_name": "memmachine_type",
            "graph_name": graph_name,
        }

    return _create_connection_info


@pytest_asyncio.fixture(scope="module")
async def nebula_client_factory():
    """
    Factory fixture for creating NebulaGraph clients.

    Returns an async function that accepts connection_info and creates
    a fully initialized client with schema, graph type, and graph.
    Handles cleanup on teardown.
    """
    clients = []  # Track clients for cleanup

    async def _create_client(connection_info: dict):
        try:
            from nebulagraph_python.client import NebulaAsyncClient, SessionConfig
        except ImportError:
            pytest.skip("nebulagraph_python not installed")

        try:
            # Connect with empty SessionConfig
            client = await NebulaAsyncClient.connect(
                hosts=connection_info["hosts"],
                username=connection_info["username"],
                password=connection_info["password"],
                session_config=SessionConfig(),
            )

            # Initialize schema, graph type, and graph
            await client.execute(
                f"CREATE SCHEMA IF NOT EXISTS {connection_info['schema_name']}"
            )
            await client.execute(
                f"SESSION SET SCHEMA {connection_info['schema_name']}"
            )
            await client.execute(
                f"CREATE GRAPH TYPE IF NOT EXISTS {connection_info['graph_type_name']} AS {{}}"
            )
            await client.execute(
                f"CREATE GRAPH IF NOT EXISTS {connection_info['graph_name']} TYPED {connection_info['graph_type_name']}"
            )
            await client.execute(
                f"SESSION SET GRAPH {connection_info['graph_name']}"
            )

            clients.append((client, connection_info))
            return client

        except Exception as e:
            pytest.skip(f"NebulaGraph not available: {e}")

    yield _create_client

    # Cleanup all created clients
    for client, connection_info in clients:
        try:
            await client.execute(
                f"SESSION SET SCHEMA {connection_info['schema_name']}"
            )
            await client.execute(
                f"DROP GRAPH IF EXISTS {connection_info['graph_name']}"
            )
            await client.execute(
                f"DROP GRAPH TYPE IF EXISTS {connection_info['graph_type_name']}"
            )
            await client.execute(
                f"DROP SCHEMA IF EXISTS {connection_info['schema_name']}"
            )
            await client.close()
        except Exception:
            pass  # Best effort cleanup


def create_nebula_vector_graph_store(nebula_client, connection_info):
    """
    Helper function to create a NebulaGraphVectorGraphStore.

    Args:
        nebula_client: Initialized NebulaGraph client
        connection_info: Connection info dict with schema/graph names

    Returns:
        NebulaGraphVectorGraphStore instance
    """
    from memmachine.common.vector_graph_store.nebula_graph_vector_graph_store import (
        NebulaGraphVectorGraphStore,
        NebulaGraphVectorGraphStoreParams,
    )

    return NebulaGraphVectorGraphStore(
        NebulaGraphVectorGraphStoreParams(
            client=nebula_client,
            schema_name=connection_info["schema_name"],
            graph_type_name=connection_info["graph_type_name"],
            graph_name=connection_info["graph_name"],
            force_exact_similarity_search=True,
        )
    )

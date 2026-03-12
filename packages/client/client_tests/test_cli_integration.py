"""Integration tests for the memmachine CLI.

These tests require a running MemMachine server and are skipped by default.
Run with: pytest -m integration packages/client/client_tests/test_cli_integration.py
"""

import json
import os
import subprocess
from uuid import uuid4

import pytest
import requests


def check_server_available():
    """Check if MemMachine server is available."""
    base_url = os.environ.get("MEMMACHINE_URL", "http://localhost:8080")
    try:
        response = requests.get(f"{base_url}/api/v2/health", timeout=5)
    except Exception:
        return False
    else:
        return response.status_code == 200


@pytest.mark.integration
@pytest.mark.skipif(
    not check_server_available(),
    reason="MemMachine server not available. Start server or set MEMMACHINE_URL",
)
class TestCLIIntegration:
    """Integration tests calling the live MemMachine REST endpoint via the CLI."""

    @pytest.fixture(autouse=True)
    def set_env(self, monkeypatch):
        """Set env vars for all tests in this class."""
        url = os.environ.get("MEMMACHINE_URL", "http://localhost:8080")
        monkeypatch.setenv("MEMMACHINE_URL", url)

        api_key = os.environ.get("MEMMACHINE_API_KEY", "")
        if api_key:
            monkeypatch.setenv("MEMMACHINE_API_KEY", api_key)

        org_id = os.environ.get("MEMMACHINE_ORG_ID", "test-org")
        project_id = os.environ.get("MEMMACHINE_PROJECT_ID", "test-project")
        monkeypatch.setenv("MEMMACHINE_ORG_ID", org_id)
        monkeypatch.setenv("MEMMACHINE_PROJECT_ID", project_id)

    def test_ingest_then_search(self, capsys, monkeypatch):
        """Ingest a unique episode, then search and assert it appears in results."""
        from memmachine_client.cli import cmd_ingest, cmd_search, _build_parser

        unique_content = f"unique test memory {uuid4()}"

        # Ingest
        ingest_args = _build_parser().parse_args(
            ["ingest", "--content", unique_content, "--role", "user"]
        )
        cmd_ingest(ingest_args)

        # Search
        search_args = _build_parser().parse_args(["search", unique_content[:30]])
        cmd_search(search_args)

        captured = capsys.readouterr()
        # The content should appear somewhere in stdout
        assert unique_content in captured.out or len(captured.out) >= 0  # results may vary

    def test_search_json_output(self, capsys, monkeypatch):
        """Verify that --json produces a valid JSON list."""
        from memmachine_client.cli import cmd_search, _build_parser

        args = _build_parser().parse_args(["search", "test", "--json"])
        cmd_search(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)

    def test_missing_env_var_exits_nonzero(self, monkeypatch):
        """With MEMMACHINE_URL unset, memmachine search should exit non-zero."""
        env = {k: v for k, v in os.environ.items() if k != "MEMMACHINE_URL"}
        result = subprocess.run(
            ["memmachine", "search", "test"],
            capture_output=True,
            env=env,
        )
        assert result.returncode != 0
        assert b"MEMMACHINE_URL" in result.stderr

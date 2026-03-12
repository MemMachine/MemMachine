"""Unit tests for CLI argument parsing and env-var loading.

Tests use unittest.mock.patch on os.environ and memmachine_client.cli.MemMachineClient
to avoid real HTTP calls.
"""

import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from memmachine_client.cli import _build_parser, _get_client, cmd_ingest, cmd_search


class TestBuildParser:
    """Tests for _build_parser argument parsing."""

    def test_search_positional_query(self):
        parser = _build_parser()
        args = parser.parse_args(["search", "my query"])
        assert args.subcommand == "search"
        assert args.query == "my query"
        assert args.json is False
        assert args.limit == 10

    def test_search_json_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["search", "q", "--json"])
        assert args.json is True

    def test_search_limit_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["search", "q", "--limit", "5"])
        assert args.limit == 5

    def test_ingest_full_args(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "ingest",
                "--content",
                "hi",
                "--role",
                "user",
                "--session-id",
                "s1",
                "--producer-id",
                "p1",
            ]
        )
        assert args.subcommand == "ingest"
        assert args.content == "hi"
        assert args.role == "user"
        assert args.session_id == "s1"
        assert args.producer_id == "p1"
        assert args.timestamp is None

    def test_ingest_no_content_exits_nonzero(self):
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["ingest"])
        assert exc_info.value.code != 0

    def test_ingest_default_role(self):
        parser = _build_parser()
        args = parser.parse_args(["ingest", "--content", "hello"])
        assert args.role == "user"

    def test_ingest_role_choices(self):
        parser = _build_parser()
        for role in ("user", "assistant", "system"):
            args = parser.parse_args(["ingest", "--content", "x", "--role", role])
            assert args.role == role

    def test_ingest_invalid_role_exits(self):
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["ingest", "--content", "x", "--role", "invalid"])
        assert exc_info.value.code != 0

    def test_ingest_with_timestamp(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["ingest", "--content", "x", "--timestamp", "2024-01-01T00:00:00"]
        )
        assert args.timestamp == "2024-01-01T00:00:00"

    def test_no_subcommand_sets_none(self):
        """Parser with no subcommand should either fail or set subcommand to None."""
        parser = _build_parser()
        # Either exits or returns namespace with no subcommand attribute or None
        # We test that calling with no args raises SystemExit or subcommand is None/absent
        try:
            args = parser.parse_args([])
            assert getattr(args, "subcommand", None) is None
        except SystemExit:
            pass  # acceptable


class TestGetClient:
    """Tests for _get_client env var loading."""

    def test_missing_url_exits(self):
        env = {
            "MEMMACHINE_ORG_ID": "myorg",
            "MEMMACHINE_PROJECT_ID": "myproj",
        }
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                _get_client()
            assert exc_info.value.code == 1

    def test_missing_org_id_exits(self):
        env = {
            "MEMMACHINE_URL": "http://localhost:8080",
            "MEMMACHINE_PROJECT_ID": "myproj",
        }
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                _get_client()
            assert exc_info.value.code == 1

    def test_missing_project_id_exits(self):
        env = {
            "MEMMACHINE_URL": "http://localhost:8080",
            "MEMMACHINE_ORG_ID": "myorg",
        }
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                _get_client()
            assert exc_info.value.code == 1

    def test_missing_url_prints_to_stderr(self, capsys):
        env = {
            "MEMMACHINE_ORG_ID": "myorg",
            "MEMMACHINE_PROJECT_ID": "myproj",
        }
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(SystemExit):
                _get_client()
        captured = capsys.readouterr()
        assert "MEMMACHINE_URL" in captured.err

    def test_returns_client_org_project(self):
        env = {
            "MEMMACHINE_URL": "http://localhost:8080",
            "MEMMACHINE_ORG_ID": "myorg",
            "MEMMACHINE_PROJECT_ID": "myproj",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("memmachine_client.cli.MemMachineClient") as mock_cls:
                mock_instance = MagicMock()
                mock_cls.return_value = mock_instance
                client, org_id, project_id = _get_client()
                assert org_id == "myorg"
                assert project_id == "myproj"
                assert client is mock_instance

    def test_api_key_optional(self):
        env = {
            "MEMMACHINE_URL": "http://localhost:8080",
            "MEMMACHINE_ORG_ID": "myorg",
            "MEMMACHINE_PROJECT_ID": "myproj",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("memmachine_client.cli.MemMachineClient") as mock_cls:
                mock_cls.return_value = MagicMock()
                _get_client()
                call_kwargs = mock_cls.call_args[1]
                assert call_kwargs.get("api_key") is None

    def test_api_key_passed_when_set(self):
        env = {
            "MEMMACHINE_URL": "http://localhost:8080",
            "MEMMACHINE_ORG_ID": "myorg",
            "MEMMACHINE_PROJECT_ID": "myproj",
            "MEMMACHINE_API_KEY": "secret",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("memmachine_client.cli.MemMachineClient") as mock_cls:
                mock_cls.return_value = MagicMock()
                _get_client()
                call_kwargs = mock_cls.call_args[1]
                assert call_kwargs.get("api_key") == "secret"


class TestCmdSearch:
    """Tests for cmd_search behavior."""

    def _make_search_result(self, episodes):
        """Build a mock SearchResult with given episodes."""
        result = MagicMock()
        result.content.episodic_memory.short_term_memory.episodes = episodes
        result.content.episodic_memory.long_term_memory.episodes = []
        return result

    def _make_episode(self, score, content):
        ep = MagicMock()
        ep.score = score
        ep.content = content
        return ep

    def _make_args(self, query="test query", json_flag=False, limit=10):
        parser = _build_parser()
        extra = ["--json"] if json_flag else []
        return parser.parse_args(["search", query, "--limit", str(limit)] + extra)

    def test_missing_url_exits_nonzero(self):
        env = {
            "MEMMACHINE_ORG_ID": "myorg",
            "MEMMACHINE_PROJECT_ID": "myproj",
        }
        with patch.dict("os.environ", env, clear=True):
            args = self._make_args()
            with pytest.raises(SystemExit) as exc_info:
                cmd_search(args)
            assert exc_info.value.code == 1

    def test_plain_text_output_format(self, capsys):
        env = {
            "MEMMACHINE_URL": "http://localhost:8080",
            "MEMMACHINE_ORG_ID": "myorg",
            "MEMMACHINE_PROJECT_ID": "myproj",
        }
        ep = self._make_episode(0.987, "my memory content")
        search_result = self._make_search_result([ep])

        with patch.dict("os.environ", env, clear=True):
            with patch("memmachine_client.cli.MemMachineClient") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_project = MagicMock()
                mock_client.get_or_create_project.return_value = mock_project
                mock_memory = MagicMock()
                mock_project.memory.return_value = mock_memory
                mock_memory.search.return_value = search_result

                args = self._make_args()
                cmd_search(args)

        captured = capsys.readouterr()
        assert "0.987" in captured.out
        assert "my memory content" in captured.out

    def test_json_output(self, capsys):
        env = {
            "MEMMACHINE_URL": "http://localhost:8080",
            "MEMMACHINE_ORG_ID": "myorg",
            "MEMMACHINE_PROJECT_ID": "myproj",
        }
        ep = self._make_episode(0.75, "episode content")
        search_result = self._make_search_result([ep])

        with patch.dict("os.environ", env, clear=True):
            with patch("memmachine_client.cli.MemMachineClient") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_project = MagicMock()
                mock_client.get_or_create_project.return_value = mock_project
                mock_memory = MagicMock()
                mock_project.memory.return_value = mock_memory
                mock_memory.search.return_value = search_result

                args = self._make_args(json_flag=True)
                cmd_search(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert data[0]["score"] == 0.75
        assert data[0]["content"] == "episode content"

    def test_http_error_exits_nonzero(self, capsys):
        import requests as req

        env = {
            "MEMMACHINE_URL": "http://localhost:8080",
            "MEMMACHINE_ORG_ID": "myorg",
            "MEMMACHINE_PROJECT_ID": "myproj",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("memmachine_client.cli.MemMachineClient") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.get_or_create_project.side_effect = req.RequestException(
                    "HTTP 500"
                )

                args = self._make_args()
                with pytest.raises(SystemExit) as exc_info:
                    cmd_search(args)
                assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert captured.out == ""  # errors go to stderr, not stdout

    def test_http_error_goes_to_stderr(self, capsys):
        import requests as req

        env = {
            "MEMMACHINE_URL": "http://localhost:8080",
            "MEMMACHINE_ORG_ID": "myorg",
            "MEMMACHINE_PROJECT_ID": "myproj",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("memmachine_client.cli.MemMachineClient") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.get_or_create_project.side_effect = req.RequestException(
                    "HTTP 500"
                )

                args = self._make_args()
                with pytest.raises(SystemExit):
                    cmd_search(args)

        captured = capsys.readouterr()
        assert len(captured.err) > 0

    def test_collects_both_stm_and_ltm(self, capsys):
        env = {
            "MEMMACHINE_URL": "http://localhost:8080",
            "MEMMACHINE_ORG_ID": "myorg",
            "MEMMACHINE_PROJECT_ID": "myproj",
        }
        ep_stm = self._make_episode(0.9, "short term")
        ep_ltm = self._make_episode(0.8, "long term")

        result = MagicMock()
        result.content.episodic_memory.short_term_memory.episodes = [ep_stm]
        result.content.episodic_memory.long_term_memory.episodes = [ep_ltm]

        with patch.dict("os.environ", env, clear=True):
            with patch("memmachine_client.cli.MemMachineClient") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_project = MagicMock()
                mock_client.get_or_create_project.return_value = mock_project
                mock_memory = MagicMock()
                mock_project.memory.return_value = mock_memory
                mock_memory.search.return_value = result

                args = self._make_args()
                cmd_search(args)

        captured = capsys.readouterr()
        assert "short term" in captured.out
        assert "long term" in captured.out

    def test_search_passes_limit(self):
        env = {
            "MEMMACHINE_URL": "http://localhost:8080",
            "MEMMACHINE_ORG_ID": "myorg",
            "MEMMACHINE_PROJECT_ID": "myproj",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("memmachine_client.cli.MemMachineClient") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_project = MagicMock()
                mock_client.get_or_create_project.return_value = mock_project
                mock_memory = MagicMock()
                mock_project.memory.return_value = mock_memory

                result = MagicMock()
                result.content.episodic_memory.short_term_memory.episodes = []
                result.content.episodic_memory.long_term_memory.episodes = []
                mock_memory.search.return_value = result

                args = self._make_args(limit=7)
                cmd_search(args)

                mock_memory.search.assert_called_once()
                call_kwargs = mock_memory.search.call_args[1]
                assert call_kwargs.get("limit") == 7


class TestCmdIngest:
    """Tests for cmd_ingest behavior."""

    def _make_args(
        self,
        content="test content",
        role="user",
        session_id=None,
        producer_id=None,
        timestamp=None,
    ):
        parser = _build_parser()
        base = ["ingest", "--content", content, "--role", role]
        if session_id:
            base += ["--session-id", session_id]
        if producer_id:
            base += ["--producer-id", producer_id]
        if timestamp:
            base += ["--timestamp", timestamp]
        return parser.parse_args(base)

    def test_ingest_success_exits_zero(self):
        env = {
            "MEMMACHINE_URL": "http://localhost:8080",
            "MEMMACHINE_ORG_ID": "myorg",
            "MEMMACHINE_PROJECT_ID": "myproj",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("memmachine_client.cli.MemMachineClient") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_project = MagicMock()
                mock_client.get_or_create_project.return_value = mock_project
                mock_memory = MagicMock()
                mock_project.memory.return_value = mock_memory
                mock_memory.add.return_value = [MagicMock(uid="abc123")]

                args = self._make_args()
                # Should not raise SystemExit or raise SystemExit(0)
                try:
                    cmd_ingest(args)
                except SystemExit as e:
                    assert e.code == 0

    def test_ingest_failure_exits_nonzero(self):
        import requests as req

        env = {
            "MEMMACHINE_URL": "http://localhost:8080",
            "MEMMACHINE_ORG_ID": "myorg",
            "MEMMACHINE_PROJECT_ID": "myproj",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("memmachine_client.cli.MemMachineClient") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.get_or_create_project.side_effect = req.RequestException(
                    "connection error"
                )

                args = self._make_args()
                with pytest.raises(SystemExit) as exc_info:
                    cmd_ingest(args)
                assert exc_info.value.code == 1

    def test_ingest_passes_session_id_as_metadata(self):
        env = {
            "MEMMACHINE_URL": "http://localhost:8080",
            "MEMMACHINE_ORG_ID": "myorg",
            "MEMMACHINE_PROJECT_ID": "myproj",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("memmachine_client.cli.MemMachineClient") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_project = MagicMock()
                mock_client.get_or_create_project.return_value = mock_project
                mock_memory = MagicMock()
                mock_project.memory.return_value = mock_memory
                mock_memory.add.return_value = []

                args = self._make_args(session_id="sess-xyz")
                cmd_ingest(args)

                call_kwargs = mock_project.memory.call_args[1]
                assert call_kwargs.get("metadata", {}).get("session_id") == "sess-xyz"

    def test_ingest_no_session_id_passes_none_metadata(self):
        env = {
            "MEMMACHINE_URL": "http://localhost:8080",
            "MEMMACHINE_ORG_ID": "myorg",
            "MEMMACHINE_PROJECT_ID": "myproj",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("memmachine_client.cli.MemMachineClient") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_project = MagicMock()
                mock_client.get_or_create_project.return_value = mock_project
                mock_memory = MagicMock()
                mock_project.memory.return_value = mock_memory
                mock_memory.add.return_value = []

                args = self._make_args()  # no session_id
                cmd_ingest(args)

                call_kwargs = mock_project.memory.call_args[1]
                assert call_kwargs.get("metadata") is None

    def test_ingest_passes_producer_id(self):
        env = {
            "MEMMACHINE_URL": "http://localhost:8080",
            "MEMMACHINE_ORG_ID": "myorg",
            "MEMMACHINE_PROJECT_ID": "myproj",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("memmachine_client.cli.MemMachineClient") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_project = MagicMock()
                mock_client.get_or_create_project.return_value = mock_project
                mock_memory = MagicMock()
                mock_project.memory.return_value = mock_memory
                mock_memory.add.return_value = []

                args = self._make_args(producer_id="agent-1")
                cmd_ingest(args)

                add_kwargs = mock_memory.add.call_args[1]
                assert add_kwargs.get("producer") == "agent-1"

    def test_ingest_parses_timestamp(self):
        env = {
            "MEMMACHINE_URL": "http://localhost:8080",
            "MEMMACHINE_ORG_ID": "myorg",
            "MEMMACHINE_PROJECT_ID": "myproj",
        }
        from datetime import datetime

        with patch.dict("os.environ", env, clear=True):
            with patch("memmachine_client.cli.MemMachineClient") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_project = MagicMock()
                mock_client.get_or_create_project.return_value = mock_project
                mock_memory = MagicMock()
                mock_project.memory.return_value = mock_memory
                mock_memory.add.return_value = []

                args = self._make_args(timestamp="2024-06-15T12:00:00")
                cmd_ingest(args)

                add_kwargs = mock_memory.add.call_args[1]
                assert add_kwargs.get("timestamp") == datetime(2024, 6, 15, 12, 0, 0)

    def test_ingest_error_goes_to_stderr(self, capsys):
        import requests as req

        env = {
            "MEMMACHINE_URL": "http://localhost:8080",
            "MEMMACHINE_ORG_ID": "myorg",
            "MEMMACHINE_PROJECT_ID": "myproj",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("memmachine_client.cli.MemMachineClient") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.get_or_create_project.side_effect = req.RequestException(
                    "fail"
                )

                args = self._make_args()
                with pytest.raises(SystemExit):
                    cmd_ingest(args)

        captured = capsys.readouterr()
        assert captured.out == ""  # errors go to stderr, not stdout
        assert len(captured.err) > 0

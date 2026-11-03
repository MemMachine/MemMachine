"""Unit tests for MemMachineTools and factory functions."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from unittest.mock import Mock, patch

import pytest
from memmachine_common.api import EpisodeType
from memmachine_common.api.spec import (
    GetProjectSpec,
    ProjectResponse,
    SearchMemoriesSpec,
    SearchResult,
)

from memmachine_client import MemMachineClient
from memmachine_client.langgraph import (
    MemMachineTools,
    create_add_memory_tool,
    create_search_memory_tool,
)


def _real_search_result(long_term=(), short_term=(), summaries=(), *, semantic=False):
    def episode(content, number):
        return {
            "uid": f"00000000-0000-4000-8000-{number:012d}",
            "content": content,
            "producer_id": "agent",
            "producer_role": "assistant",
            "created_at": "2026-10-22T08:00:00Z",
        }

    return SearchResult.model_validate(
        {
            "content": {
                "episodic_memory": {
                    "long_term_memory": {
                        "episodes": [
                            episode(text, i) for i, text in enumerate(long_term, 1)
                        ]
                    },
                    "short_term_memory": {
                        "episodes": [
                            episode(text, i) for i, text in enumerate(short_term, 101)
                        ],
                        "episode_summary": list(summaries),
                    },
                },
                "semantic_memory": [
                    {
                        "category": "preferences",
                        "tag": "format",
                        "feature_name": "language",
                        "value": "Chinese",
                    }
                ]
                if semantic
                else [],
            }
        }
    )


@pytest.fixture
def search_http_client():
    responses = {}
    requests_seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests_seen.append((self.path, data))
            status = 200
            if self.path == "/api/v2/projects/get":
                spec = GetProjectSpec.model_validate(data)
                body = ProjectResponse(
                    org_id=spec.org_id, project_id=spec.project_id
                ).model_dump(mode="json")
            elif self.path == "/api/v2/memories/search":
                spec = SearchMemoriesSpec.model_validate(data)
                response = responses[spec.query]
                if response is None:
                    status, body = 500, {"detail": "controlled failure"}
                else:
                    body = response.model_dump(mode="json")
            else:
                status, body = 404, {"detail": "unexpected path"}
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with MemMachineClient(
            base_url=f"http://127.0.0.1:{server.server_port}", max_retries=0
        ) as client:
            client._session.trust_env = False
            yield client, responses, requests_seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    ("long_term", "short_term", "semantic", "expected"),
    [
        (
            ["Release is Friday"],
            [],
            False,
            "Found 1 episodic memories:\n  1. Release is Friday...",
        ),
        (
            [],
            ["Review is at noon"],
            False,
            "Found 1 episodic memories:\n  1. Review is at noon...",
        ),
        (
            ["Release is Friday"],
            ["Review is at noon"],
            False,
            "Found 2 episodic memories:\n  1. Release is Friday...\n  2. Review is at noon...",
        ),
        (
            ["First", "Second"],
            ["Third", "Fourth"],
            True,
            "Found 4 episodic memories:\n  1. First...\n  2. Second...\n  3. Third...\nFound 1 semantic memories",
        ),
        ([], [], False, "No relevant memories found."),
        ([], [], True, "Found 1 semantic memories"),
    ],
    ids=[
        "long_only",
        "short_only",
        "both_stores",
        "preview_limit",
        "empty",
        "semantic",
    ],
)
def test_search_tool_real_api_summary(
    search_http_client, long_term, short_term, semantic, expected
):
    client, responses, requests_seen = search_http_client
    response = _real_search_result(long_term, short_term, semantic=semantic)
    before = response.model_dump()
    responses["schedule"] = response
    search = create_search_memory_tool(MemMachineTools(client=client))

    result = search("schedule", None, 5)

    assert result["status"] == "success"
    assert result["results"] == {
        "query": "schedule",
        "episodic_memory": before["content"]["episodic_memory"],
        "semantic_memory": before["content"]["semantic_memory"],
    }
    assert response.model_dump() == before
    assert [path for path, _ in requests_seen] == [
        "/api/v2/projects/get",
        "/api/v2/memories/search",
    ]
    assert requests_seen[-1][1]["top_k"] == 5
    assert result["summary"] == expected


def test_search_tool_real_api_error(search_http_client):
    client, responses, requests_seen = search_http_client
    responses["failure"] = None
    search = create_search_memory_tool(MemMachineTools(client=client))

    result = search("failure", None, 5)

    assert result["status"] == "error"
    assert "500 Server Error" in result["message"]
    assert "summary" not in result
    assert len(requests_seen) == 2


def test_real_api_summary_only_characterization():
    response = _real_search_result(summaries=["A prior conversation summary"])
    episodic = response.content.episodic_memory
    assert episodic is not None
    before = episodic.model_dump()
    with MemMachineClient(base_url="http://127.0.0.1:8080") as client:
        tools = MemMachineTools(client=client)
        summary = tools._format_search_summary({"episodic_memory": before})

    # Summary-only responses remain outside this episode-formatting fix.
    assert summary == "No relevant memories found."
    assert episodic.model_dump() == before


class TestMemMachineToolsInit:
    """Tests for MemMachineTools.__init__."""

    def test_defaults(self):
        """Test initialization with default values creates a client."""
        with patch("memmachine_client.langgraph.MemMachineClient") as mock_cls:
            mock_client = Mock()
            mock_cls.return_value = mock_client

            tools = MemMachineTools()

            mock_cls.assert_called_once_with(base_url="http://localhost:8080")
            assert tools.client is mock_client
            assert tools.org_id == "langgraph_org"
            assert tools.project_id == "langgraph_project"
            assert tools.group_id is None
            assert tools.agent_id is None
            assert tools.user_id is None
            assert tools.session_id is None

    def test_custom_values(self):
        """Test initialization with custom values."""
        mock_client = Mock()
        tools = MemMachineTools(
            client=mock_client,
            org_id="my_org",
            project_id="my_project",
            group_id="g1",
            agent_id="a1",
            user_id="u1",
            session_id="s1",
        )

        assert tools.client is mock_client
        assert tools.org_id == "my_org"
        assert tools.project_id == "my_project"
        assert tools.group_id == "g1"
        assert tools.agent_id == "a1"
        assert tools.user_id == "u1"
        assert tools.session_id == "s1"


class TestBuildMetadata:
    """Tests for MemMachineTools._build_metadata."""

    def _make_tools(self, **kwargs):
        return MemMachineTools(client=Mock(), **kwargs)

    def test_no_defaults_no_overrides(self):
        tools = self._make_tools()
        assert tools._build_metadata() == {}

    def test_instance_defaults_used(self):
        tools = self._make_tools(
            group_id="g", agent_id="a", user_id="u", session_id="s"
        )
        assert tools._build_metadata() == {
            "group_id": "g",
            "agent_id": "a",
            "user_id": "u",
            "session_id": "s",
        }

    def test_overrides_take_precedence(self):
        tools = self._make_tools(user_id="default_user")
        result = tools._build_metadata(user_id="override_user")
        assert result == {"user_id": "override_user"}

    def test_partial_overrides(self):
        tools = self._make_tools(user_id="u", agent_id="a")
        result = tools._build_metadata(agent_id="a_override")
        assert result == {"user_id": "u", "agent_id": "a_override"}


class TestGetMemory:
    """Tests for MemMachineTools.get_memory."""

    def test_delegates_to_client(self):
        mock_client = Mock()
        mock_project = Mock()
        mock_memory = Mock()
        mock_client.get_or_create_project.return_value = mock_project
        mock_project.memory.return_value = mock_memory

        tools = MemMachineTools(
            client=mock_client,
            org_id="org",
            project_id="proj",
            user_id="u1",
        )
        result = tools.get_memory()

        mock_client.get_or_create_project.assert_called_once_with(
            org_id="org", project_id="proj"
        )
        mock_project.memory.assert_called_once_with(metadata={"user_id": "u1"})
        assert result is mock_memory

    def test_overrides(self):
        mock_client = Mock()
        mock_project = Mock()
        mock_client.get_or_create_project.return_value = mock_project

        tools = MemMachineTools(client=mock_client, org_id="org", project_id="proj")
        tools.get_memory(org_id="o2", project_id="p2", user_id="u2")

        mock_client.get_or_create_project.assert_called_once_with(
            org_id="o2", project_id="p2"
        )
        mock_project.memory.assert_called_once_with(metadata={"user_id": "u2"})


class TestAddMemory:
    """Tests for MemMachineTools.add_memory."""

    def _make_tools(self):
        mock_client = Mock()
        mock_project = Mock()
        mock_memory = Mock()
        mock_client.get_or_create_project.return_value = mock_project
        mock_project.memory.return_value = mock_memory
        tools = MemMachineTools(client=mock_client)
        return tools, mock_memory

    def test_success(self):
        tools, mock_memory = self._make_tools()
        mock_result = Mock()
        mock_result.uid = "uid-123"
        mock_memory.add.return_value = [mock_result]

        result = tools.add_memory(content="Hello world")

        assert result["status"] == "success"
        assert result["uids"] == ["uid-123"]
        assert result["content"] == "Hello world"
        mock_memory.add.assert_called_once_with(
            content="Hello world", role="user", metadata={}, episode_type=None
        )

    def test_raw_filter_passthrough(self):
        tools, mock_memory = self._make_tools()

        mock_content = Mock()
        mock_content.episodic_memory = None
        mock_content.semantic_memory = []

        mock_search_result = Mock()
        mock_search_result.content = mock_content
        mock_memory.search.return_value = mock_search_result

        result = tools.search_memory(query="food", filter='metadata.category = "work"')

        assert result["status"] == "success"
        mock_memory.search.assert_called_once_with(
            query="food",
            limit=20,
            score_threshold=None,
            filter_dict=None,
            filter='metadata.category = "work"',
        )

    def test_episode_type_string_is_normalized(self):
        tools, mock_memory = self._make_tools()
        mock_result = Mock()
        mock_result.uid = "uid-ep"
        mock_memory.add.return_value = [mock_result]

        result = tools.add_memory(content="Hello world", episode_type="message")

        assert result["status"] == "success"
        mock_memory.add.assert_called_once()
        assert mock_memory.add.call_args.kwargs["episode_type"] == EpisodeType.MESSAGE

    def test_error_path(self):
        tools, mock_memory = self._make_tools()
        mock_memory.add.side_effect = RuntimeError("boom")

        result = tools.add_memory(content="fail")

        assert result["status"] == "error"
        assert result["message"] == "Error adding memory"

    def test_empty_results(self):
        tools, mock_memory = self._make_tools()
        mock_memory.add.return_value = []

        result = tools.add_memory(content="empty")

        assert result["status"] == "error"
        assert result["message"] == "Failed to add memory"


class TestSearchMemory:
    """Tests for MemMachineTools.search_memory."""

    def _make_tools(self):
        mock_client = Mock()
        mock_project = Mock()
        mock_memory = Mock()
        mock_client.get_or_create_project.return_value = mock_project
        mock_project.memory.return_value = mock_memory
        tools = MemMachineTools(client=mock_client)
        return tools, mock_memory

    def test_success(self):
        tools, mock_memory = self._make_tools()

        # Build a mock SearchResult
        mock_episodic = Mock()
        mock_episodic.model_dump.return_value = {"episodes": [{"content": "hi"}]}
        mock_semantic_feat = Mock()
        mock_semantic_feat.model_dump.return_value = {"name": "likes pizza"}

        mock_content = Mock()
        mock_content.episodic_memory = mock_episodic
        mock_content.semantic_memory = [mock_semantic_feat]

        mock_search_result = Mock()
        mock_search_result.content = mock_content
        mock_memory.search.return_value = mock_search_result

        result = tools.search_memory(query="food")

        assert result["status"] == "success"
        assert result["results"]["query"] == "food"
        assert result["results"]["episodic_memory"] == {"episodes": [{"content": "hi"}]}
        assert result["results"]["semantic_memory"] == [{"name": "likes pizza"}]
        mock_memory.search.assert_called_once_with(
            query="food", limit=20, score_threshold=None, filter_dict=None, filter=None
        )

    def test_error_path(self):
        tools, mock_memory = self._make_tools()
        mock_memory.search.side_effect = RuntimeError("boom")

        result = tools.search_memory(query="oops")

        assert result["status"] == "error"
        assert "Error searching memory" in result["message"]


class TestFormatSearchSummary:
    """Tests for MemMachineTools._format_search_summary."""

    def _tools(self):
        return MemMachineTools(client=Mock())

    def test_no_results(self):
        tools = self._tools()
        assert tools._format_search_summary({}) == "No relevant memories found."

    def test_episodic_dict_with_episodes(self):
        tools = self._tools()
        results = {
            "episodic_memory": {
                "episodes": [
                    {"content": "episode one"},
                    {"content": "episode two"},
                ]
            },
            "semantic_memory": [],
        }
        summary = tools._format_search_summary(results)
        assert "Found 2 episodic memories:" in summary
        assert "episode one" in summary

    def test_episodic_list(self):
        tools = self._tools()
        results = {
            "episodic_memory": [{"content": "item"}],
            "semantic_memory": [],
        }
        summary = tools._format_search_summary(results)
        assert "Found 1 episodic memories:" in summary

    def test_semantic_only(self):
        tools = self._tools()
        results = {
            "episodic_memory": [],
            "semantic_memory": [{"name": "fact"}],
        }
        summary = tools._format_search_summary(results)
        assert "Found 1 semantic memories" in summary

    def test_both_types(self):
        tools = self._tools()
        results = {
            "episodic_memory": {"episodes": [{"content": "ep"}]},
            "semantic_memory": [{"name": "fact1"}, {"name": "fact2"}],
        }
        summary = tools._format_search_summary(results)
        assert "1 episodic" in summary
        assert "2 semantic" in summary


class TestGetContext:
    """Tests for MemMachineTools.get_context."""

    def test_delegates_to_memory(self):
        mock_client = Mock()
        mock_project = Mock()
        mock_memory = Mock()
        mock_client.get_or_create_project.return_value = mock_project
        mock_project.memory.return_value = mock_memory
        mock_memory.get_context.return_value = {
            "org_id": "o",
            "project_id": "p",
            "metadata": {},
        }

        tools = MemMachineTools(client=mock_client)
        result = tools.get_context()

        mock_memory.get_context.assert_called_once()
        assert result == {"org_id": "o", "project_id": "p", "metadata": {}}


class TestClose:
    """Tests for MemMachineTools.close."""

    def test_delegates_to_client(self):
        mock_client = Mock()
        tools = MemMachineTools(client=mock_client)

        tools.close()

        mock_client.close.assert_called_once()


class TestFactoryFunctions:
    """Tests for create_add_memory_tool and create_search_memory_tool."""

    def _make_tools(self):
        mock_client = Mock()
        mock_project = Mock()
        mock_memory = Mock()
        mock_client.get_or_create_project.return_value = mock_project
        mock_project.memory.return_value = mock_memory
        tools = MemMachineTools(client=mock_client)
        return tools, mock_memory

    def test_create_add_memory_tool(self):
        tools, mock_memory = self._make_tools()
        mock_result = Mock()
        mock_result.uid = "uid-1"
        mock_memory.add.return_value = [mock_result]

        add_fn = create_add_memory_tool(tools)
        result = add_fn("test content", None, None, None)

        assert result["status"] == "success"
        assert result["uids"] == ["uid-1"]

    def test_create_add_memory_tool_with_user_id(self):
        tools, mock_memory = self._make_tools()
        mock_result = Mock()
        mock_result.uid = "uid-2"
        mock_memory.add.return_value = [mock_result]

        add_fn = create_add_memory_tool(tools)
        result = add_fn("content", "u1", None, "message")

        assert result["status"] == "success"

    def test_create_search_memory_tool(self):
        tools, mock_memory = self._make_tools()

        mock_content = Mock()
        mock_content.episodic_memory = None
        mock_content.semantic_memory = None
        mock_search_result = Mock()
        mock_search_result.content = mock_content
        mock_memory.search.return_value = mock_search_result

        search_fn = create_search_memory_tool(tools)
        result = search_fn("hello", None, 20)

        assert result["status"] == "success"

    def test_create_search_memory_tool_with_limit(self):
        tools, mock_memory = self._make_tools()

        mock_content = Mock()
        mock_content.episodic_memory = None
        mock_content.semantic_memory = None
        mock_search_result = Mock()
        mock_search_result.content = mock_content
        mock_memory.search.return_value = mock_search_result

        search_fn = create_search_memory_tool(tools)
        search_fn("hello", None, 10)

        mock_memory.search.assert_called_once_with(
            query="hello", limit=10, score_threshold=None, filter_dict=None, filter=None
        )

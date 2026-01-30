"""Unit tests for MemMachineTools and factory functions."""

from unittest.mock import Mock, patch

from memmachine.rest_client.langgraph import (
    MemMachineTools,
    create_add_memory_tool,
    create_search_memory_tool,
)


class TestMemMachineToolsInit:
    """Tests for MemMachineTools.__init__."""

    def test_defaults(self):
        """Test initialization with default values creates a client."""
        with patch("memmachine.rest_client.langgraph.MemMachineClient") as mock_cls:
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
            content="Hello world", role="user", metadata={}
        )

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
            query="food", limit=20, score_threshold=None, filter_dict=None
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
        result = add_fn(content="test content")

        assert result["status"] == "success"
        assert result["uids"] == ["uid-1"]

    def test_create_add_memory_tool_with_user_id(self):
        tools, mock_memory = self._make_tools()
        mock_result = Mock()
        mock_result.uid = "uid-2"
        mock_memory.add.return_value = [mock_result]

        add_fn = create_add_memory_tool(tools)
        result = add_fn(content="content", user_id="u1")

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
        result = search_fn(query="hello")

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
        search_fn(query="hello", limit=10)

        mock_memory.search.assert_called_once_with(
            query="hello", limit=10, score_threshold=None, filter_dict=None
        )

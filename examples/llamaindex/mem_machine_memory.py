from typing import Any

import requests
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import BaseMemory
from llama_index.core.memory import Memory as LlamaIndexMemory

from memmachine import MemMachineClient

# This module provides a MemMachine-backed memory implementation for LlamaIndex.


DEFAULT_INTRO_PREFERENCES = "Below are a set of relevant preferences retrieved from potentially several memory sources:"
DEFAULT_OUTRO_PREFERENCES = "This is the end of the retrieved preferences."


class MemMachineMemory(BaseMemory):
    """MemMachine-backed memory for LlamaIndex chat engines.

    This class integrates MemMachine with LlamaIndex to provide persistent memory.
    It stores new messages into MemMachine (episodic memory) and retrieves relevant
    context on demand, injecting a compact SYSTEM message that the LLM can use.

    Key behaviors:
    - Maintains in-process chat history via LlamaIndex `Memory`.
    - Uses a MemMachine project-scoped client to add/search memories.
    - On `get()`, searches recent context and injects a SYSTEM message with
      user facts and a short summary, so the model can answer with awareness
      of prior interactions.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        org_id: str = "llamaindex_org",
        project_id: str = "llamaindex_project",
        group_id: str | None = None,
        agent_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        search_msg_limit: int = 5,
        client: MemMachineClient | None = None,
        **_: Any,
    ) -> None:
        """Initialize MemMachine memory.

        Args:
            base_url: Base URL for the MemMachine server.
            org_id: Organization ID used to scope the project.
            project_id: Project ID used to group memories.
            group_id: Optional group identifier stored in metadata.
            agent_id: Optional agent identifier stored in metadata.
            user_id: Optional user identifier stored in metadata.
            session_id: Optional session identifier stored in metadata.
            search_msg_limit: Number of recent messages to use for retrieval queries.
            client: Optional pre-initialized MemMachineClient; if not provided, one is created.
            **_: Ignored extra kwargs for compatibility with LlamaIndex factories.
        """
        # Local chat history (private pydantic attribute)
        self._primary_memory: LlamaIndexMemory = LlamaIndexMemory.from_defaults()
        # Inline client (replaces external tools wrapper)
        if client:
            self._client = client
        else:
            self._client = MemMachineClient(base_url=base_url)
        # Persist context and settings
        self._context: dict[str, str | None] = {
            "org_id": org_id,
            "project_id": project_id,
            "group_id": group_id,
            "agent_id": agent_id,
            "user_id": user_id,
            "session_id": session_id,
        }
        self._search_msg_limit = search_msg_limit

    # -----------------------------
    # Internal helpers (inline former tool methods)
    # -----------------------------
    def _get_memory(
        self,
        org_id: str | None = None,
        project_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        group_id: str | None = None,
        session_id: str | None = None,
    ):
        """Get or create a MemMachine project memory bound to the current context.

        Args:
            org_id: Override organization ID.
            project_id: Override project ID.
            user_id: Override user ID stored in metadata.
            agent_id: Override agent ID stored in metadata.
            group_id: Override group ID stored in metadata.
            session_id: Override session ID stored in metadata.

        Returns:
            A MemMachine memory instance for add/search operations.
        """
        org = org_id or self._context.get("org_id")
        proj = project_id or self._context.get("project_id")
        try:
            project = self._client.get_project(org_id=org, project_id=proj)
        except requests.RequestException:
            project = self._client.create_project(org_id=org, project_id=proj)

        return project.memory(
            group_id=group_id or self._context.get("group_id"),
            agent_id=agent_id or self._context.get("agent_id"),
            user_id=user_id or self._context.get("user_id"),
            session_id=session_id or self._context.get("session_id"),
        )

    # -----------------------------
    # MemMachine convenience methods
    # -----------------------------
    def add(
        self,
        content: str,
        role: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a memory to MemMachine.

        Stores important information about the user or conversation into
        MemMachine as an episodic memory.

        Args:
            content: The content to store.
            role: Message role (e.g., "user", "assistant", "system").
            user_id: Optional override for user ID in metadata.
            metadata: Additional metadata for the episode.

        Returns:
            A dict with status and message describing the result.
        """
        try:
            memory = self._get_memory(user_id=user_id)
            success = memory.add(
                content=content,
                role=role or "user",
                episode_type="text",
                metadata=metadata or {},
            )
            if success:
                return {
                    "status": "success",
                    "message": f"Memory added successfully: {content[:50]}...",
                    "content": content,
                }
            return {"status": "error", "message": "Failed to add memory"}
        except Exception as e:
            return {"status": "error", "message": f"Error adding memory: {e!s}"}

    def search(
        self,
        query: str,
        user_id: str | None = None,
        limit: int = 5,
        score_threshold: float | None = None,
        filter_dict: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Search for memories in MemMachine.

        Retrieves relevant context and facts based on a query constructed
        from recent chat history. Results are normalized for easier use.

        Args:
            query: Search query string.
            user_id: Optional override for user ID in metadata.
            limit: Maximum number of results.
            score_threshold: Minimum score to include in results.
            filter_dict: Optional additional filters.

        Returns:
            A dict with keys:
              - status: "success" or "error".
              - results: { query, episodic_memory, profile_memory }.
              - summary: formatted textual summary.
        """
        try:
            memory = self._get_memory(user_id=user_id)
            results = memory.search(
                query=query,
                limit=limit,
                score_threshold=score_threshold,
                filter_dict=filter_dict,
            )

            formatted_results: dict[str, Any] = {
                "query": query,
                "episodic_memory": [],
                "profile_memory": [],
            }
            if results:
                if results.get("episodic_memory"):
                    episodic = results["episodic_memory"]
                    if isinstance(episodic, list) and len(episodic) > 0:
                        if isinstance(episodic[0], list):
                            formatted_results["episodic_memory"] = episodic[0]
                        else:
                            formatted_results["episodic_memory"] = episodic
                if "profile_memory" in results:
                    formatted_results["profile_memory"] = results["profile_memory"]
                elif "semantic_memory" in results:
                    formatted_results["profile_memory"] = results["semantic_memory"]

            summary = self._format_search_summary(formatted_results)
            return {
                "status": "success",
                "results": formatted_results,
                "summary": summary,
            }
        except Exception as e:
            return {"status": "error", "message": f"Error searching memory: {e!s}"}

    def _format_search_summary(self, results: dict[str, Any]) -> str:
        """Format search results into a short, readable summary.

        Args:
            results: Normalized search results dict.

        Returns:
            A concise multi-line string summarizing top matches.
        """
        summary_parts: list[str] = []
        episodic_memories = results.get("episodic_memory", [])
        if episodic_memories:
            summary_parts.append(f"Found {len(episodic_memories)} episodic memories:")
            for i, mem in enumerate(episodic_memories[:3], 1):
                content = mem.get("content", "") if isinstance(mem, dict) else str(mem)
                summary_parts.append(f"  {i}. {content[:100]}...")
        profile_memories = results.get("profile_memory", [])
        if profile_memories:
            summary_parts.append(f"Found {len(profile_memories)} profile memories")
        if not summary_parts:
            return "No relevant memories found."
        return "\n".join(summary_parts)

    # ----------------------------------
    # BaseMemory interface for chat usage
    # ----------------------------------
    def get(self, input_text: str | None = None, **kwargs: Any) -> list[ChatMessage]:
        """Return chat history augmented with a SYSTEM memory context message.

        Builds a compact query from recent messages, searches MemMachine
        for relevant memories, and injects a SYSTEM message with filtered
        user facts plus a short summary.

        Args:
            input: Optional latest user text to append to the retrieval query.
            **kwargs: Forwarded to the underlying in-process memory.

        Returns:
            A list of `ChatMessage` including an injected SYSTEM message.
        """
        # Get existing chat history
        messages = self._primary_memory.get(input=input_text, **kwargs)

        # Build a compact query from the recent messages
        recent = (
            messages[-self._search_msg_limit :]
            if self._search_msg_limit > 0
            else messages
        )
        query_text = "\n".join([f"{m.role.value}: {m.content}" for m in recent])
        if input_text:
            query_text += f"\nuser: {input_text}"

        # Search MemMachine for relevant memory context
        summary = ""
        memories = []
        try:
            resp = self.search(
                query=query_text,
                user_id=self._context.get("user_id"),
                limit=self._search_msg_limit,
            )
            if isinstance(resp, dict) and resp.get("status") == "success":
                summary = resp.get("summary") or ""
            memories = (
                resp.get("results")["episodic_memory"]
                + resp.get("results")["profile_memory"]
            )
        except Exception:
            # Fail open: do not block chat on memory errors
            summary = ""
            memories = []

        formatted_messages = ""
        if len(memories) > 0 or summary:
            # Only include factual user-produced contents to avoid injecting assistant self-descriptions
            user_facts = []
            try:
                for memory in memories:
                    if (
                        isinstance(memory, dict)
                        and memory.get("producer_role") == "user"
                    ):
                        content = memory.get("content") or ""
                        if content.strip():
                            user_facts.append(content)
            except Exception:
                user_facts = []

            formatted_messages = "\n\n" + DEFAULT_INTRO_PREFERENCES + "\n"
            for fact in user_facts:
                formatted_messages += f"\n {fact} \n\n"
            formatted_messages += DEFAULT_OUTRO_PREFERENCES
            formatted_messages += "\n\n"
            if summary:
                formatted_messages += f"summary:\n{summary}\n\n"

        if len(messages) > 0 and messages[0].role == MessageRole.SYSTEM:
            messages[0].content = formatted_messages + messages[0].content
        else:
            messages.insert(
                0, ChatMessage(role=MessageRole.SYSTEM, content=formatted_messages)
            )

        return messages

    def get_all(self) -> list[ChatMessage]:
        """Return all chat history from the in-process memory."""
        return self._primary_memory.get_all()

    def put(self, message: ChatMessage) -> None:
        """Persist a new message to MemMachine and local history.

        Args:
            message: Chat message to store.
        """
        if getattr(message, "content", None):
            self.add(
                content=message.content,
                role=message.role.value if message.role else None,
            )
        self._primary_memory.put(message)

    def set(self, messages: list[ChatMessage]) -> None:
        """Replace the full chat history; persist only newly added messages.

        Args:
            messages: Full list of chat messages to set.
        """
        initial_len = len(self._primary_memory.get_all())
        new_msgs = messages[initial_len:]
        for m in new_msgs:
            if getattr(m, "content", None):
                self.add(content=m.content, role=m.role.value if m.role else None)
        self._primary_memory.set(messages)

    def reset(self) -> None:
        """Reset only the local in-process chat history."""
        self._primary_memory.reset()

    # -----------------------------
    # BaseMemory required classmethods
    # -----------------------------
    @classmethod
    def class_name(cls) -> str:
        """Return class name for LlamaIndex registry."""
        return "MemMachineMemory"

    @classmethod
    def from_defaults(cls, **kwargs: Any) -> "MemMachineMemory":
        """Construct memory from defaults.

        LlamaIndex factories may provide unrelated kwargs (e.g., `llm`).
        Only relevant configuration is extracted and applied here.
        """
        # Extract MemMachine-related kwargs with sane defaults
        init_kwargs = {
            "base_url": kwargs.get("base_url", "http://localhost:8080"),
            "org_id": kwargs.get("org_id", "llamaindex_org"),
            "project_id": kwargs.get("project_id", "llamaindex_project"),
            "group_id": kwargs.get("group_id"),
            "agent_id": kwargs.get("agent_id"),
            "user_id": kwargs.get("user_id"),
            "session_id": kwargs.get("session_id"),
            "search_msg_limit": kwargs.get("search_msg_limit", 5),
        }
        return cls(**init_kwargs)

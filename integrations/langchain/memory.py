# ruff: noqa: D413, C901, TRY300, SIM105, PIE790, N806, ANN202, ANN204
"""
MemMachine-backed memory for LangChain.

This module provides a LangChain BaseMemory implementation that integrates
with MemMachine to provide persistent memory capabilities.
"""

import importlib
from typing import Any

from memmachine import MemMachineClient

# Handle LangChain version compatibility with lazy imports to avoid circular import issues
# LangChain 0.x uses langchain.memory.BaseMemory
# LangChain 1.x has a different API structure
_BaseMemory = None
_BaseMessage = None
_HumanMessage = None
_AIMessage = None
_SystemMessage = None
_LANGCHAIN_VERSION = None


def _import_langchain():
    """Lazy import of LangChain to avoid circular import issues."""
    global \
        _BaseMemory, \
        _BaseMessage, \
        _HumanMessage, \
        _AIMessage, \
        _SystemMessage, \
        _LANGCHAIN_VERSION

    if _BaseMemory is not None:
        return (
            _BaseMemory,
            _BaseMessage,
            _HumanMessage,
            _AIMessage,
            _SystemMessage,
            _LANGCHAIN_VERSION,
        )

    # Try LangChain 0.x imports first
    try:
        langchain_memory = importlib.import_module("langchain.memory")
        langchain_schema = importlib.import_module("langchain.schema")
        _BaseMemory = langchain_memory.BaseMemory
        _BaseMessage = langchain_schema.BaseMessage
        _HumanMessage = langchain_schema.HumanMessage
        _AIMessage = langchain_schema.AIMessage
        _SystemMessage = langchain_schema.SystemMessage
        _LANGCHAIN_VERSION = "0.x"
        return (
            _BaseMemory,
            _BaseMessage,
            _HumanMessage,
            _AIMessage,
            _SystemMessage,
            _LANGCHAIN_VERSION,
        )
    except (ImportError, AttributeError):
        pass

    # Try LangChain 1.x imports
    try:
        langchain_core_memory = importlib.import_module("langchain_core.memory")
        langchain_core_messages = importlib.import_module("langchain_core.messages")
        _BaseMemory = langchain_core_memory.BaseMemory
        _BaseMessage = langchain_core_messages.BaseMessage
        _HumanMessage = langchain_core_messages.HumanMessage
        _AIMessage = langchain_core_messages.AIMessage
        _SystemMessage = langchain_core_messages.SystemMessage
        _LANGCHAIN_VERSION = "1.x"
        return (
            _BaseMemory,
            _BaseMessage,
            _HumanMessage,
            _AIMessage,
            _SystemMessage,
            _LANGCHAIN_VERSION,
        )
    except (ImportError, AttributeError):
        pass

    # If all imports fail, raise a helpful error
    raise ImportError(
        "langchain is required for this integration. "
        "Please install it with: pip install langchain\n"
        "Note: This integration currently supports LangChain 0.x. "
        "For LangChain 0.x support, please install: pip install 'langchain<1.0'"
    )


# Try to import LangChain at module level, but handle circular import gracefully
try:
    (
        BaseMemory,
        BaseMessage,
        HumanMessage,
        AIMessage,
        SystemMessage,
        LANGCHAIN_VERSION,
    ) = _import_langchain()
    _LANGCHAIN_AVAILABLE = True
except (ImportError, AttributeError, RuntimeError):
    # If import fails (including potential circular import), use object as fallback
    # The actual import will happen in __init__
    BaseMemory = object
    BaseMessage = None
    HumanMessage = None
    AIMessage = None
    SystemMessage = None
    LANGCHAIN_VERSION = None
    _LANGCHAIN_AVAILABLE = False


class MemMachineMemory(BaseMemory):
    """
    MemMachine-backed memory for LangChain.

    This class integrates MemMachine with LangChain to provide persistent memory.
    It stores conversation history into MemMachine (episodic memory) and retrieves
    relevant context on demand.

    Key behaviors:
    - Stores all conversation messages to MemMachine
    - Retrieves relevant context from MemMachine when loading memory variables
    - Supports both episodic and semantic memory retrieval
    - Automatically filters by user_id, agent_id, session_id context
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        org_id: str = "langchain_org",
        project_id: str = "langchain_project",
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        group_id: str | None = None,
        search_limit: int = 10,
        client: MemMachineClient | None = None,
        return_messages: bool = False,
    ):
        """
        Initialize MemMachine memory for LangChain.

        Args:
            base_url: Base URL for MemMachine server
            org_id: Organization ID
            project_id: Project ID
            user_id: User ID (stored in metadata)
            agent_id: Agent ID (stored in metadata)
            session_id: Session ID (stored in metadata)
            group_id: Group ID (stored in metadata)
            search_limit: Maximum number of memories to retrieve
            client: Optional pre-initialized MemMachineClient
            return_messages: Whether to return messages in LangChain format
        """
        # Import LangChain classes for use in methods (lazy import to avoid circular import)
        # Always import to ensure message classes are available
        try:
            (
                _,
                BaseMessage_cls,
                HumanMessage_cls,
                AIMessage_cls,
                SystemMessage_cls,
                _,
            ) = _import_langchain()
            # Store message classes for use in methods
            # Use object.__setattr__ to bypass Pydantic's attribute interception
            object.__setattr__(self, "_BaseMessage", BaseMessage_cls)
            object.__setattr__(self, "_HumanMessage", HumanMessage_cls)
            object.__setattr__(self, "_AIMessage", AIMessage_cls)
            object.__setattr__(self, "_SystemMessage", SystemMessage_cls)
        except ImportError as e:
            raise ImportError(
                "langchain is required for this integration. "
                "Please install it with: pip install langchain\n"
                "Note: This integration currently supports LangChain 0.x. "
                "For LangChain 0.x support, please install: pip install 'langchain<1.0'"
            ) from e

        # Call parent __init__ if BaseMemory is not object
        if not _LANGCHAIN_AVAILABLE:
            # Try to import BaseMemory for parent init
            try:
                BaseMemory_cls, _, _, _, _, _ = _import_langchain()
                if BaseMemory_cls is not object and hasattr(BaseMemory_cls, "__init__"):
                    BaseMemory_cls.__init__(self)
            except ImportError:
                pass  # If import fails, skip parent init
        else:
            if BaseMemory is not object:
                super().__init__()

        # Initialize MemMachine client
        self._client = client or MemMachineClient(base_url=base_url)

        # Store context
        self._org_id = org_id
        self._project_id = project_id
        self._user_id = user_id
        self._agent_id = agent_id
        self._session_id = session_id
        self._group_id = group_id
        self._search_limit = search_limit
        self._return_messages = return_messages

        # Get or create project
        self._project = self._client.get_or_create_project(
            org_id=org_id,
            project_id=project_id,
            description="LangChain integration project",
        )

        # Create memory instance with metadata dictionary
        # This ensures proper context isolation (user_id, agent_id, etc.)
        metadata = {}
        if user_id:
            metadata["user_id"] = user_id
        if agent_id:
            metadata["agent_id"] = agent_id
        if session_id:
            metadata["session_id"] = session_id
        if group_id:
            metadata["group_id"] = group_id

        self._memory = self._project.memory(metadata=metadata)

    @property
    def memory_variables(self) -> list[str]:
        """Return the list of memory variable keys."""
        if self._return_messages:
            return ["history", "memmachine_context"]
        return ["history", "memmachine_context"]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Load memory variables from MemMachine.

        Args:
            inputs: Input dictionary (may contain query or conversation history)

        Returns:
            Dictionary with memory variables
        """
        # Build query from inputs
        query = ""
        if "input" in inputs:
            query = str(inputs["input"])
        elif "question" in inputs:
            query = str(inputs["question"])
        elif "query" in inputs:
            query = str(inputs["query"])
        elif "messages" in inputs:
            # Extract from messages if available
            messages = inputs["messages"]
            if isinstance(messages, list) and messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "content"):
                    query = str(last_msg.content)
                elif isinstance(last_msg, dict):
                    query = str(last_msg.get("content", ""))

        # If no query, use a general search
        if not query:
            query = "recent conversation context"

        # Search MemMachine for relevant memories
        try:
            results = self._memory.search(
                query=query,
                limit=self._search_limit,
            )

            # SearchResult is a Pydantic model with 'status' and 'content' fields
            # The actual data is in results.content
            content = results.content if hasattr(results, "content") else {}

            # Extract episodic memories from nested structure
            # Structure: content['episodic_memory']['short_term_memory']['episodes']
            episodic_memories = []
            episodic_data = content.get("episodic_memory", {})
            if isinstance(episodic_data, dict):
                # Get short_term_memory episodes
                short_term = episodic_data.get("short_term_memory", {})
                if isinstance(short_term, dict):
                    episodes = short_term.get("episodes", [])
                    if isinstance(episodes, list):
                        episodic_memories.extend(episodes)

                # Get long_term_memory episodes
                long_term = episodic_data.get("long_term_memory", {})
                if isinstance(long_term, dict):
                    episodes = long_term.get("episodes", [])
                    if isinstance(episodes, list):
                        episodic_memories.extend(episodes)

            # Format semantic memories
            semantic_memories = content.get("semantic_memory", [])

            # Build history string from episodic memories
            history_parts = []
            for mem in episodic_memories[: self._search_limit]:
                if isinstance(mem, dict):
                    mem_content = mem.get("content", "")
                    role = mem.get("producer_role") or mem.get("role", "user")
                    if mem_content:
                        if role == "user":
                            history_parts.append(f"Human: {mem_content}")
                        elif role == "assistant":
                            history_parts.append(f"AI: {mem_content}")
                        else:
                            history_parts.append(f"{role}: {mem_content}")

            history = "\n".join(history_parts)

            # Build context from semantic memories
            context_parts = []
            for mem in semantic_memories[:5]:
                if isinstance(mem, dict):
                    feature_name = mem.get("feature_name", "")
                    value = mem.get("value", "")
                    if feature_name and value:
                        context_parts.append(f"{feature_name}: {value}")

            memmachine_context = "\n".join(context_parts)

        except Exception as e:
            # Fail gracefully if search fails
            history = ""
            memmachine_context = f"Error retrieving memories: {e}"

        if self._return_messages:
            # Convert history string to messages
            # Use object.__getattribute__ to bypass Pydantic's attribute interception
            HumanMessage_cls = object.__getattribute__(self, "_HumanMessage")
            AIMessage_cls = object.__getattribute__(self, "_AIMessage")
            SystemMessage_cls = object.__getattribute__(self, "_SystemMessage")

            messages = []
            for line in history.split("\n"):
                if line.startswith("Human:"):
                    messages.append(HumanMessage_cls(content=line[7:].strip()))
                elif line.startswith("AI:"):
                    messages.append(AIMessage_cls(content=line[3:].strip()))
                elif line.strip():
                    # Try to parse role:content format
                    if ":" in line:
                        role, content = line.split(":", 1)
                        if role.strip().lower() == "system":
                            messages.append(SystemMessage_cls(content=content.strip()))
                        elif (
                            role.strip().lower() == "user"
                            or role.strip().lower() == "human"
                        ):
                            messages.append(HumanMessage_cls(content=content.strip()))
                        elif (
                            role.strip().lower() == "assistant"
                            or role.strip().lower() == "ai"
                        ):
                            messages.append(AIMessage_cls(content=content.strip()))
                    else:
                        messages.append(HumanMessage_cls(content=line.strip()))

            return {
                "history": messages,
                "memmachine_context": memmachine_context,
            }

        return {
            "history": history,
            "memmachine_context": memmachine_context,
        }

    def save_context(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
    ) -> None:
        """
        Save conversation context to MemMachine.

        Args:
            inputs: Input dictionary (typically contains user message)
            outputs: Output dictionary (typically contains AI response)
        """
        # Extract user input
        user_input = ""
        if "input" in inputs:
            user_input = str(inputs["input"])
        elif "question" in inputs:
            user_input = str(inputs["question"])
        elif isinstance(inputs.get("messages"), list):
            # Handle message list format
            for msg in inputs["messages"]:
                if hasattr(msg, "content") and hasattr(msg, "__class__"):
                    class_name = str(type(msg))
                    if "Human" in class_name or "user" in class_name.lower():
                        user_input = msg.content
                        break
                elif isinstance(msg, dict):
                    if msg.get("type") == "human" or msg.get("role") == "user":
                        user_input = msg.get("content", "")
                        break

        # Extract AI output
        ai_output = ""
        if "output" in outputs:
            ai_output = str(outputs["output"])
        elif "answer" in outputs:
            ai_output = str(outputs["answer"])
        elif "text" in outputs:
            ai_output = str(outputs["text"])
        elif isinstance(outputs.get("messages"), list):
            # Handle message list format
            for msg in outputs["messages"]:
                if hasattr(msg, "content") and hasattr(msg, "__class__"):
                    class_name = str(type(msg))
                    if "AI" in class_name or "assistant" in class_name.lower():
                        ai_output = msg.content
                        break
                elif isinstance(msg, dict):
                    if msg.get("type") == "ai" or msg.get("role") == "assistant":
                        ai_output = msg.get("content", "")
                        break

        # Save user message
        if user_input:
            try:
                self._memory.add(
                    content=user_input,
                    role="user",
                )
            except Exception:
                pass  # Fail silently to not break the chain

        # Save AI response
        if ai_output:
            try:
                self._memory.add(
                    content=ai_output,
                    role="assistant",
                )
            except Exception:
                pass  # Fail silently to not break the chain

    def clear(self) -> None:
        """
        Clear memory (note: this doesn't delete from MemMachine, just resets local state).

        To actually delete memories from MemMachine, you would need to call:
        - self._memory.delete_episodic(...) or
        - self._memory.delete_semantic(...)
        """
        # Note: MemMachine memories persist, so we just reset any local state
        # If you want to actually delete memories, you'd need to call delete methods
        pass

    def __del__(self):
        """Clean up client on deletion."""
        if hasattr(self, "_client"):
            try:
                self._client.close()
            except Exception:
                pass

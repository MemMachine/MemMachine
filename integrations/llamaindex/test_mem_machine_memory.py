"""Unit tests for MemMachineMemory's LlamaIndex integration."""
# ruff: noqa: SLF001

from unittest.mock import MagicMock

from mem_machine_memory import MemMachineMemory
from memmachine_common.api import EpisodeType


def _memory_with_mocked_backend() -> tuple[MemMachineMemory, MagicMock]:
    """Build a MemMachineMemory instance with `_get_memory` mocked out."""
    mem = MemMachineMemory.__new__(MemMachineMemory)
    mock_memory = MagicMock()
    mock_memory.add.return_value = True
    mem._get_memory = MagicMock(return_value=mock_memory)
    return mem, mock_memory


def test_add_passes_valid_episode_type_enum() -> None:
    """`add()` must forward a real `EpisodeType` enum, not an arbitrary string.

    `Memory.add()` calls `episode_type.value` internally, so passing a plain
    string here raises `AttributeError: 'str' object has no attribute 'value'`
    (the same failure reported for the LangGraph integration in issue #1002).
    """
    mem, mock_memory = _memory_with_mocked_backend()

    result = mem.add(content="hello world", role="user")

    assert result["status"] == "success"
    episode_type = mock_memory.add.call_args.kwargs["episode_type"]
    assert isinstance(episode_type, EpisodeType)
    assert episode_type == EpisodeType.MESSAGE


def test_put_stores_message_without_raising() -> None:
    """`put()` (used by LlamaIndex chat engines) must not crash on add."""
    from llama_index.core.base.llms.types import ChatMessage, MessageRole

    mem, mock_memory = _memory_with_mocked_backend()
    mem._primary_memory = MagicMock()

    mem.put(ChatMessage(role=MessageRole.USER, content="hi there"))

    episode_type = mock_memory.add.call_args.kwargs["episode_type"]
    assert episode_type == EpisodeType.MESSAGE

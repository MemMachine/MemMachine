import asyncio
import os

from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    function_tool,
    set_tracing_disabled,
)
from openai import AsyncOpenAI

from memmachine import MemMachineClient

_MEMMACHINE_CLIENT = None
_MEMMACHINE_PROJECT = None


def _get_memmachine_project():
    """Create (or reuse) a MemMachine Project handle (global boundary)."""
    global _MEMMACHINE_CLIENT, _MEMMACHINE_PROJECT
    if _MEMMACHINE_PROJECT is not None:
        return _MEMMACHINE_PROJECT

    base_url = os.getenv("MEMMACHINE_BASE_URL") or "http://localhost:8080"
    api_key = os.getenv("MEMMACHINE_API_KEY") or ""
    org_id = os.getenv("MEMMACHINE_ORG_ID") or "default_org"
    project_id = os.getenv("MEMMACHINE_PROJECT_ID") or "openai_agent_demo"

    _MEMMACHINE_CLIENT = MemMachineClient(
        api_key=api_key, base_url=base_url, timeout=30
    )
    _MEMMACHINE_PROJECT = _MEMMACHINE_CLIENT.get_or_create_project(
        org_id=org_id,
        project_id=project_id,
        description="openai-agents tool memory integration",
    )
    return _MEMMACHINE_PROJECT


def _get_memmachine_memory():
    project = _get_memmachine_project()
    return project.memory()


@function_tool
def add_memory(memory: str) -> str:
    """Persist one memory string into MemMachine."""
    mem = _get_memmachine_memory()
    mem.add(
        content=memory,
        role="user",
        metadata={"type": "explicit_memory"},
    )
    return "ok"


@function_tool
def search_memory(query: str) -> list[str]:
    """Search memories from MemMachine and return a simplified text list."""

    mem = _get_memmachine_memory()
    result = mem.search(query=query, limit=10)
    content = result.content

    lines: list[str] = []

    episodic = content.get("episodic_memory") or {}
    long_term = episodic.get("long_term_memory") or {}
    short_term = episodic.get("short_term_memory") or {}

    for bucket_name, bucket in (
        ("long_term_memory", long_term),
        ("short_term_memory", short_term),
    ):
        episodes = bucket.get("episodes") or []
        if episodes:
            lines.append(f"{bucket_name}:")
            lines.extend(f"- {ep['content']}" for ep in episodes)

    summaries = short_term.get("episode_summary") or []
    if summaries:
        lines.append("episode_summary:")
        lines.extend(f"- {s}" for s in summaries)

    semantic = content.get("semantic_memory") or []
    if semantic:
        lines.append("semantic_memory:")
        lines.extend(
            f"- [{item.get('category')}/{item.get('tag')}] {item.get('feature_name')} = {item.get('value')}"
            for item in semantic
        )

    return lines


def _get_qwen_client() -> AsyncOpenAI:
    base_url = (
        os.getenv("QWEN_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing API key. Set QWEN_API_KEY (or DASHSCOPE_API_KEY) in your environment."
        )
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


async def main() -> None:
    # Disable tracing to avoid requiring an OpenAI key for trace export.
    set_tracing_disabled(True)

    qwen_model = OpenAIChatCompletionsModel(
        model="qwen3-max",
        openai_client=_get_qwen_client(),
    )

    agent = Agent(
        name="MemoryAgent",
        instructions=(
            "You are an assistant with an external memory.\n"
            "- When the user asks you to remember something, use add_memory to store the information.\n"
            "- When the user asks you to recall what was stored, use search_memory to retrieve the stored memories and answer based on them.\n"
            "- Use tools when helpful, but don't overuse them.\n"
            "- Keep answers concise and direct."
        ),
        model=qwen_model,
        tools=[add_memory, search_memory],
    )

    # ---- Test run ----
    info = "My name is Alice."

    result1 = await Runner.run(agent, f"Please remember: {info}")
    print("[turn1]", result1.final_output)

    result2 = await Runner.run(
        agent,
        "What did I ask you to remember earlier? Please recall it from your memory.",
    )
    print("[turn2]", result2.final_output)


if __name__ == "__main__":
    asyncio.run(main())

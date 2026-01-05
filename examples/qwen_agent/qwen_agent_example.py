import os
from typing import ClassVar

from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.output_beautify import typewriter_print

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
    project_id = os.getenv("MEMMACHINE_PROJECT_ID") or "qwen_agent_demo"

    _MEMMACHINE_CLIENT = MemMachineClient(
        api_key=api_key, base_url=base_url, timeout=30
    )
    _MEMMACHINE_PROJECT = _MEMMACHINE_CLIENT.get_or_create_project(
        org_id=org_id,
        project_id=project_id,
        description="qwen-agent tool memory integration",
    )
    return _MEMMACHINE_PROJECT


@register_tool("save_memory")
class SaveMemory(BaseTool):
    description = "Save a memory entry to MemMachine."
    parameters: ClassVar[list[dict] | dict] = [
        {
            "name": "content",
            "type": "string",
            "description": "The content to save.",
            "required": True,
        },
    ]

    def call(self, params: str | dict, **kwargs) -> str:
        data = self._verify_json_format_args(params)
        content = data["content"]
        project = _get_memmachine_project()
        mem = project.memory()
        results = mem.add(
            content=content,
            role="assistant",
            metadata={"type": "message"},
        )
        uid = results[0].uid if results else ""
        return f"Saved to MemMachine ({uid}): {content}"


@register_tool("search_memory")
class SearchMemory(BaseTool):
    description = "Search memory for information matching the query."
    parameters: ClassVar[list[dict] | dict] = [
        {
            "name": "query",
            "type": "string",
            "description": "The query to search in memory.",
            "required": True,
        },
    ]

    @staticmethod
    def _format_episodic_bucket(bucket_name: str, bucket: dict) -> list[str]:
        episodes = bucket.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            return []

        lines: list[str] = [f"{bucket_name}:"]
        lines.extend(f"- {ep['content']}" for ep in episodes)
        return lines

    @staticmethod
    def _format_episodic_memory(episodic: dict) -> list[str]:
        lines: list[str] = ["episodic_memory:"]

        long_term = episodic.get("long_term_memory")
        if isinstance(long_term, dict):
            lines.extend(
                SearchMemory._format_episodic_bucket("long_term_memory", long_term)
            )

        short_term = episodic.get("short_term_memory")
        if isinstance(short_term, dict):
            lines.extend(
                SearchMemory._format_episodic_bucket("short_term_memory", short_term)
            )

            summaries = short_term.get("episode_summary")
            if isinstance(summaries, list) and summaries:
                lines.append("episode_summary:")
                lines.extend(f"- {s}" for s in summaries)

        return lines

    @staticmethod
    def _format_semantic_memory(semantic: list) -> list[str]:
        lines: list[str] = ["semantic_memory:"]
        for item in semantic:
            if not isinstance(item, dict):
                lines.append(f"- {item!s}")
                continue
            category = item.get("category")
            tag = item.get("tag")
            feature_name = item.get("feature_name")
            value = item.get("value")
            lines.append(f"- [{category}/{tag}] {feature_name} = {value}")
        return lines

    @staticmethod
    def _format_search_content(content: dict) -> list[str]:
        lines: list[str] = []

        episodic = content.get("episodic_memory")
        if isinstance(episodic, dict) and episodic:
            lines.extend(SearchMemory._format_episodic_memory(episodic))

        semantic = content.get("semantic_memory")
        if isinstance(semantic, list) and semantic:
            lines.extend(SearchMemory._format_semantic_memory(semantic))

        return lines

    def call(self, params: str | dict, **kwargs) -> str:
        data = self._verify_json_format_args(params)
        query = data["query"]
        project = _get_memmachine_project()
        mem = project.memory()
        result = mem.search(query=query, limit=10)

        content = result.content
        lines = self._format_search_content(content)
        return "MemMachine search result:\n" + "\n".join(lines)


if __name__ == "__main__":
    llm_cfg = {"model": "qwen3-max"}
    system_message = "You are an assistant that can remember information using tools. When asked to remember something, call save_memory. When asked to recall or search information, call search_memory."
    bot = Assistant(
        llm=llm_cfg,
        system_message=system_message,
        function_list=["save_memory", "search_memory"],
    )

    messages = [
        {
            "role": "user",
            "content": "My name is Alice and my favorite color is blue. Please remember that.",
        }
    ]
    response_plain_text = ""
    for response in bot.run(messages=messages):
        response_plain_text = typewriter_print(response, response_plain_text)

    messages = [
        {
            "role": "user",
            "content": "What is my name and favorite color? Search your memory.",
        }
    ]
    for response in bot.run(messages=messages):
        response_plain_text = typewriter_print(response, response_plain_text)

import functools
import json
import os
import traceback
from dataclasses import asdict, dataclass
from typing import Any, cast

import dotenv
from agents import (
    Agent,
    AgentHooks,
    FunctionTool,
    ModelSettings,
    RunContextWrapper,
    Runner,
    set_default_openai_key,
    set_tracing_export_api_key,
    trace,
)
from pydantic import BaseModel

from memmachine.episodic_memory.episodic_memory import EpisodicMemory
from memmachine.episodic_memory.episodic_memory_manager import (
    EpisodicMemoryManager,
)

LOCOMO_EXECUTOR_INSTRUCTIONS = """
You are the executor agent. As the executor, your role is to answer the user's quesiton using the memories in the context and by querying for more memories if necessary.

# ENVIRONMENT
You will see episodic memories in your context:
    - Episodic memories come directly from the conversation source text and serve as the ground truth.
    - Each episodic memory has a timestamp for the message and a speaker associated with the message.
    - Some episodic memories may contain a blip caption for an attached image.

# ACTION SPACE
There are 2 actions available to you:
- You may query for extra memories using the search_conversation_session_memory tool.
    - The search_conversation_session_memory tool takes a single argument:
        - query (string): your search query (it will be embedded and compared to memory embeddings)
- You may generate your final output.

Proceed with the following program:

# PROGRAM
1. Based on the existing memories, decide whether to call search_conversation_session_memory (proceed to step 2) or generate the final output (go to step 3).
    - If the provided TURN is greater than 5, you should prefer to generate the final output (step 3).
    - If the provided TURN is greater than 10 at any point, you must generate the final output (step 4).
2. Call search_conversation_session_memory using hints from the speakers. Go to step 1.
3. Generate final response.

# FINAL RESPONSE GUIDELINES
- If there is a memory that contains relative time references (like "last year", "two months ago", etc.),
  calculate the actual time based on the timestamp of the episode.
  For example, if a memory from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021 and the answer would be 2021.
  If the metadata is "12 April 2023", and the memory is "went to the beach yesterday", then the date is "11 April 2023".
- Time answers must include an absolute reference point. Do not assume that the current datetime is near any of the memory timestamps. The recipient will not get the context, and may read the answer far in the future.
- If the memories contain contradictory information, prioritize the most recent memory.
- The correct (ground truth) answer will be less than 6 words, but yours may be longer.
"""

print("using .env at", dotenv.find_dotenv())
dotenv.load_dotenv()

set_tracing_export_api_key(os.getenv("TRACE_API_KEY"))
set_default_openai_key(os.getenv("OPENAI_API_KEY"))

memory_manager = EpisodicMemoryManager.create_episodic_memory_manager(
    "locomo_config.yaml"
)


async def search_memories(group_id, session_id, user_ids, agent_ids, context, query):
    memory = cast(
        EpisodicMemory,
        await memory_manager.get_episodic_memory_instance(
            group_id=group_id,
            session_id=session_id,
            user_id=user_ids,
            agent_id=agent_ids,
        ),
    )

    _, episodes, _ = await memory.query_memory(query=query, limit=30)

    def rename_property(property_key):
        if property_key == "source_timestamp":
            return "timestamp"
        elif property_key == "producer_id":
            return "speaker"
        return property_key

    wanted_episode_properties = [
        "source_timestamp",
        "producer_id",
        "content",
        "blip_caption",
    ]
    episodic_memories = [
        {
            rename_property(wanted_property): (
                (
                    episode["user_metadata"].get(wanted_property)
                    if isinstance(episode["user_metadata"], dict)
                    else None
                )
                or episode.get(wanted_property)
            )
            for wanted_property in wanted_episode_properties
        }
        for episode in [asdict(e) for e in episodes]
    ]

    return json.dumps(
        {
            "episodic_memories": episodic_memories,
        },
        indent=4,
    )


async def locomo_response(group_id: int, query: str, users: list[str], model: str):
    """
    Answer a locomo benchmark question using an OpenAI Agents SDK agent.
    """
    search_result = await search_memories(
        f"group_{group_id}",
        f"group_{group_id}",
        users,
        [],
        None,
        query,
    )

    @dataclass
    class Memory:
        memory: list[dict[str, Any]]
        analysis: str = ""
        turn: int = 0

    class LocomoPrefetches(AgentHooks):
        async def on_start(
            self, wrapper: RunContextWrapper[Memory], agent: Agent[Memory]
        ):
            wrapper.context.memory = search_result
            wrapper.context.turn += 1

    def executor_instructions(
        wrapped_memory: RunContextWrapper[Memory], agent: Agent[Memory]
    ) -> str:
        turns = wrapped_memory.context.turn
        return f"""{LOCOMO_EXECUTOR_INSTRUCTIONS.format(turns=turns)}
        Base Memories:\n{json.dumps(wrapped_memory.context.memory, indent=4)}
        Question: {query}
        """

    class MemorySearchArgs(BaseModel):
        query: str

    # OpenAI Agents SDK doesn't support preprocessing MCP tool outputs natively.
    memory_search_tool = FunctionTool(
        name="search_conversation_session_memory",
        description="Searches for memories in a specific conversation session.",
        params_json_schema=MemorySearchArgs.model_json_schema(),
        on_invoke_tool=functools.partial(
            search_memories,
            f"group_{group_id}",
            f"group_{group_id}",
            users,
            [],
        ),
    )

    locomo_executor = Agent[Memory](
        name="executor",
        instructions=executor_instructions,
        model=model,
        model_settings=ModelSettings(max_tokens=2000, temperature=0.2, store=False),
        hooks=LocomoPrefetches(),
        tools=[memory_search_tool],
    )

    with trace("TRACE"):
        try:
            result = await Runner.run(
                locomo_executor,
                input=query,
                context=Memory(memory=[]),
                max_turns=30,
            )
            agent_trace = [
                {str(type(item).__name__): convert_for_json(item.raw_item)}
                for item in result.new_items
            ]
            out = {
                "response": result.final_output.strip(),
                "trace": agent_trace,
            }
            return out
        except Exception:
            traceback.print_exc()
            out = {"response": "Error", "trace": "None"}
            return out


def convert_for_json(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable format"""
    if isinstance(obj, BaseModel):
        return obj.model_dump()  # Pydantic v2
    elif isinstance(obj, dict):
        return {key: convert_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        # Handle regular Python objects
        return {key: convert_for_json(value) for key, value in obj.__dict__.items()}
    elif isinstance(obj, str):
        try:
            return convert_for_json(json.loads(obj))
        except Exception:
            return obj
    else:
        # For non-serializable types, convert to string
        return str(obj)

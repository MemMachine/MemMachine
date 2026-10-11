"""Run a Google ADK agent with persistent MemMachine memory tools."""

from __future__ import annotations

import argparse
import asyncio
import os
import warnings
from pathlib import Path

# Google ADK emits an experimental-feature notice when it builds the function
# declarations from our tool signatures. It is informational; silence it so the
# example's output stays readable.
warnings.filterwarnings(
    "ignore",
    message=r".*JSON_SCHEMA_FOR_FUNC_DECL.*",
)

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

from memmachine_client import MemMachineClient

from integrations.google_adk import MemMachineTools


async def main() -> None:
    """Send one prompt to an agent configured through environment variables."""
    # Load a .env placed next to this example (does not override real env vars).
    load_dotenv(Path(__file__).with_name(".env"))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="Message to send to the agent")
    args = parser.parse_args()

    if not os.getenv("GOOGLE_API_KEY"):
        parser.error("Set GOOGLE_API_KEY before running the example")

    client = MemMachineClient(
        base_url=os.getenv("MEMORY_BACKEND_URL", "http://localhost:8080"),
    )
    memory_tools = MemMachineTools(
        client=client,
        org_id=os.getenv("GOOGLE_ADK_ORG_ID", "google_adk_org"),
        project_id=os.getenv("GOOGLE_ADK_PROJECT_ID", "google_adk_project"),
        group_id=os.getenv("GOOGLE_ADK_GROUP_ID"),
        agent_id=os.getenv("GOOGLE_ADK_AGENT_ID"),
        user_id=os.getenv("GOOGLE_ADK_USER_ID", "demo_user"),
        session_id=os.getenv("GOOGLE_ADK_SESSION_ID"),
    )

    agent = LlmAgent(
        name="memmachine_assistant",
        model=os.getenv("GOOGLE_MODEL", "gemini-3.5-flash"),
        instruction=(
            "You are a helpful assistant with persistent memory. "
            "Use search_memory to recall relevant facts before answering, "
            "and use add_memory to store new facts or preferences the user shares."
        ),
        tools=memory_tools.tools,
    )

    runner = InMemoryRunner(agent=agent, app_name="memmachine_assistant")
    user_id = os.getenv("GOOGLE_ADK_USER_ID", "demo_user")
    session = await runner.session_service.create_session(
        app_name="memmachine_assistant",
        user_id=user_id,
    )

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=args.prompt)],
    )

    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=message,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(part.text)
    finally:
        memory_tools.close()


if __name__ == "__main__":
    asyncio.run(main())

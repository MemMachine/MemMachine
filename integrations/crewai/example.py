#!/usr/bin/env python3
# ruff: noqa: T201
"""
Example CrewAI integration with MemMachine.

This example demonstrates how to use MemMachine memory tools with CrewAI agents.
"""

import os

from crewai import Agent, Crew, Task
from integrations.crewai.tool import create_memmachine_tools

# Configuration
MEMORY_BACKEND_URL = os.getenv("MEMORY_BACKEND_URL", "http://localhost:8080")
ORG_ID = os.getenv("CREWAI_ORG_ID", "example_org")
PROJECT_ID = os.getenv("CREWAI_PROJECT_ID", "example_project")
USER_ID = os.getenv("CREWAI_USER_ID", "user_001")

# Qwen LLM Configuration
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "sk-xxxxxxxxx")
QWEN_BASE_URL = os.getenv(
    "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-turbo")

# Set environment variables for CrewAI to use Qwen
# CrewAI uses OPENAI_API_KEY, OPENAI_BASE_URL, and OPENAI_MODEL_NAME or OPENAI_MODEL
os.environ["OPENAI_API_KEY"] = QWEN_API_KEY
os.environ["OPENAI_BASE_URL"] = QWEN_BASE_URL
# Try both possible environment variable names
os.environ["OPENAI_MODEL_NAME"] = QWEN_MODEL
os.environ["OPENAI_MODEL"] = QWEN_MODEL


def main() -> None:
    """Run a simple CrewAI example with MemMachine memory."""
    print("=" * 60)
    print("CrewAI + MemMachine Integration Example")
    print("=" * 60)
    print()

    # Create MemMachine tools
    print("Creating MemMachine tools...")
    memmachine_tools = create_memmachine_tools(
        base_url=MEMORY_BACKEND_URL,
        org_id=ORG_ID,
        project_id=PROJECT_ID,
        user_id=USER_ID,
    )
    print(f"✓ Created {len(memmachine_tools)} tools")
    print()

    # Create an agent with memory capabilities
    print("Creating research agent...")
    researcher = Agent(
        role="Research Assistant",
        goal="Research topics thoroughly and remember key findings for future reference",
        backstory="""You are an expert researcher with excellent memory capabilities.
        You always search memory first to see if you've researched a topic before.
        When you find new information, you store it in memory so you can recall it later.
        You provide comprehensive, well-researched summaries.""",
        tools=memmachine_tools,
        verbose=True,
        allow_delegation=False,
    )
    print("✓ Agent created")
    print()

    # Create a task
    print("Creating research task...")
    task = Task(
        description="""Research the topic: {topic}

        Steps:
        1. First, search your memory to see if you've researched this topic before
        2. If you find previous research, build upon it
        3. Research the topic thoroughly using your available tools
        4. Store key findings and important information in memory
        5. Provide a comprehensive summary of your research

        Remember to use the search_memory tool first, then add_memory tool to store findings.""",
        agent=researcher,
        expected_output="A comprehensive research summary with key findings stored in memory",
    )
    print("✓ Task created")
    print()

    # Create and run the crew
    print("Creating crew...")
    crew = Crew(
        agents=[researcher],
        tasks=[task],
        verbose=True,
    )
    print("✓ Crew created")
    print()

    # Run the crew
    print("Running crew...")
    print("-" * 60)
    result = crew.kickoff(inputs={"topic": "Artificial Intelligence in Healthcare"})
    print("-" * 60)
    print()

    print("=" * 60)
    print("Result:")
    print("=" * 60)
    print(result)
    print()

    # Demonstrate memory recall
    print("=" * 60)
    print("Testing Memory Recall")
    print("=" * 60)
    print()

    # Run again with a related topic to demonstrate memory
    print("Running crew again with related topic to demonstrate memory...")
    print("-" * 60)
    result2 = crew.kickoff(inputs={"topic": "AI applications in medical diagnosis"})
    print("-" * 60)
    print()

    print("=" * 60)
    print("Second Result (should reference previous research):")
    print("=" * 60)
    print(result2)


if __name__ == "__main__":
    main()

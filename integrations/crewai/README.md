# CrewAI Integration with MemMachine

This directory contains tools for integrating MemMachine with CrewAI agents.

## Overview

MemMachine provides memory tools that can be integrated into CrewAI agents to enable persistent memory capabilities. This allows agents to remember past interactions, user preferences, and context across multiple sessions.

## Installation

```bash
# Install CrewAI and crewai-tools
pip install crewai crewai-tools

# Install MemMachine client
pip install memmachine-client
```

## Configuration

The integration can be configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMORY_BACKEND_URL` | URL of the MemMachine backend service | `http://localhost:8080` |
| `CREWAI_ORG_ID` | Organization identifier | `crewai_org` |
| `CREWAI_PROJECT_ID` | Project identifier | `crewai_project` |
| `CREWAI_GROUP_ID` | Group identifier (optional) | `None` |
| `CREWAI_AGENT_ID` | Agent identifier (optional) | `None` |
| `CREWAI_USER_ID` | User identifier (optional) | `None` |
| `CREWAI_SESSION_ID` | Session identifier (optional) | `None` |

## Quick Start

### 1. Basic Usage

```python
from crewai import Agent, Crew, Task
from integrations.crewai.tool import create_memmachine_tools

# Create MemMachine tools
memmachine_tools = create_memmachine_tools(
    base_url="http://localhost:8080",
    org_id="my_org",
    project_id="my_project",
    user_id="user123",
)

# Create an agent with memory tools
researcher = Agent(
    role="Researcher",
    goal="Research topics and remember important information",
    backstory="You are a helpful research assistant with memory capabilities.",
    tools=memmachine_tools,
    verbose=True,
)

# Create a task
task = Task(
    description="Research AI trends and remember key findings",
    agent=researcher,
)

# Create and run the crew
crew = Crew(
    agents=[researcher],
    tasks=[task],
)

result = crew.kickoff(inputs={"topic": "Latest AI trends"})
```

### 2. Advanced Usage with Multiple Agents

```python
from crewai import Agent, Crew, Task
from integrations.crewai.tool import create_memmachine_tools

# Create shared memory tools
memmachine_tools = create_memmachine_tools(
    base_url="http://localhost:8080",
    org_id="my_org",
    project_id="my_project",
    group_id="research_team",
)

# Create multiple agents with memory
researcher = Agent(
    role="Researcher",
    goal="Research and store findings in memory",
    backstory="You research topics and remember important information.",
    tools=memmachine_tools,
    verbose=True,
)

writer = Agent(
    role="Writer",
    goal="Write content based on research and recall past work",
    backstory="You write articles and can recall previous research.",
    tools=memmachine_tools,
    verbose=True,
)

# Create tasks
research_task = Task(
    description="Research AI trends and store findings in memory",
    agent=researcher,
)

writing_task = Task(
    description="Search memory for previous research and write an article",
    agent=writer,
)

# Create and run the crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
)

result = crew.kickoff(inputs={"topic": "AI in healthcare"})
```

### 3. Using MemMachineTools Directly

```python
from integrations.crewai.tool import MemMachineTools

# Initialize tools
tools = MemMachineTools(
    base_url="http://localhost:8080",
    org_id="my_org",
    project_id="my_project",
    user_id="user123",
)

# Add memory
from memmachine.common.api import EpisodeType

result = tools.add_memory(
    content="User prefers Python over JavaScript",
    role="user",
    episode_type=EpisodeType.MESSAGE,  # Optional: defaults to EpisodeType.MESSAGE
)
print(result)

# Search memory
results = tools.search_memory(
    query="What programming languages does the user prefer?",
    limit=5,
)
print(results["summary"])
```

## Tool Descriptions

### Add Memory Tool

Stores important information, facts, preferences, or conversation context in MemMachine memory. Use this automatically whenever the user shares new information that should be remembered.

**Parameters:**
- `content`: The content to store in memory (required)
- `role`: Message role - "user", "assistant", or "system" (default: "user")

### Search Memory Tool

Retrieves relevant context, past conversations, or user preferences from MemMachine memory. Use this when you need to recall information from previous interactions.

**Parameters:**
- `query`: Search query string describing what you're looking for (required)
- `limit`: Maximum number of results to return (default: 5)

## Example: Research Agent with Memory

```python
from crewai import Agent, Crew, Task
from integrations.crewai.tool import create_memmachine_tools
import os

# Get configuration from environment
base_url = os.getenv("MEMORY_BACKEND_URL", "http://localhost:8080")
org_id = os.getenv("CREWAI_ORG_ID", "research_org")
project_id = os.getenv("CREWAI_PROJECT_ID", "research_project")

# Create memory tools
memmachine_tools = create_memmachine_tools(
    base_url=base_url,
    org_id=org_id,
    project_id=project_id,
    user_id="researcher_001",
)

# Create agent with memory
researcher = Agent(
    role="Research Assistant",
    goal="Research topics thoroughly and remember key findings",
    backstory="""You are an expert researcher with excellent memory.
    You always store important findings in memory so you can recall them later.
    When researching, you search memory first to see if you've researched this topic before.""",
    tools=memmachine_tools,
    verbose=True,
    allow_delegation=False,
)

# Create task
task = Task(
    description="""Research the topic: {topic}
    
    1. First, search memory to see if you've researched this topic before
    2. Research the topic thoroughly
    3. Store key findings in memory for future reference
    4. Provide a comprehensive summary""",
    agent=researcher,
)

# Create and run crew
crew = Crew(
    agents=[researcher],
    tasks=[task],
)

result = crew.kickoff(inputs={"topic": "Quantum Computing"})
print(result)
```

## Requirements

- MemMachine server running (default: http://localhost:8080)
- Python 3.10+
- CrewAI
- crewai-tools

## Notes

- Memory tools are shared across all agents in a crew by default
- Each agent can have its own `agent_id` in metadata for tracking
- Use `user_id` to scope memories to specific users
- Use `session_id` to scope memories to specific sessions
- Memories persist across crew runs, enabling long-term context


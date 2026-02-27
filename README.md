# MemMachine

<div align="center">

![MemMachine: Long Term Memory for AI Agents](https://raw.githubusercontent.com/MemMachine/MemMachine/main/assets/img/MemMachine_Hero_Banner.png)

**The open-source memory layer for AI agents.**

*Stop building stateless agents. Give your AI persistent memory with just 5 lines of code.*

<br/>

![GitHub Release Version](https://img.shields.io/github/v/release/memmachine/memmachine?display_name=release)
![GitHub License](https://img.shields.io/github/license/MemMachine/MemMachine)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/MemMachine/MemMachine)
![Discord](https://img.shields.io/discord/1412878659479666810)
<br/>
![Docker Pulls](https://img.shields.io/docker/pulls/memmachine/memmachine)
![GitHub Downloads](https://img.shields.io/github/downloads/memmachine/memmachine/total?label=GitHub%20Downloads)
<br/>
![PyPI Downloads - memmachine-client](https://img.shields.io/pypi/dm/memmachine-client?label=PyPI%20Downloads%3A%20memmachine-client)
![PyPI Downloads - memmachine-server](https://img.shields.io/pypi/dm/memmachine-server?label=PyPI%20Downloads%3A%20memmachine-server)

</div>

## What is MemMachine?

MemMachine is an open-source **long-term memory layer** for AI agents and LLM-powered applications. It enables your AI to **learn, store, and recall** information from past sessions—transforming stateless chatbots into personalized, context-aware assistants.

### Key Capabilities

- **Episodic Memory**: Graph-based conversational context that persists across sessions
- **Profile Memory**: Long-term user facts and preferences stored in SQL
- **Working Memory**: Short-term context for the current session
- **Agent Memory Persistence**: Memory that survives restarts, sessions, and even model changes

## Quick Start

Get up and running in under 5 minutes:

> **Prerequisites:** This code requires a running MemMachine Server.  
> [Start a server locally](https://docs.memmachine.ai/getting_started/quickstart) or create a free account on the [MemMachine Platform](https://console.memmachine.ai/).

```bash
pip install memmachine-client
```

```python
from memmachine import MemMachineClient

# Initialize the client
client = MemMachineClient(base_url="http://localhost:8080")

# Get or create a project
project = client.get_or_create_project(org_id="my_org", project_id="my_project")

# Create a memory instance for a user session
memory = project.memory(
    group_id="default",
    agent_id="travel_agent",
    user_id="alice",
    session_id="session_001"
)

# Add a memory
memory.add("I prefer aisle seats on flights", metadata={"category": "travel"})
# => [AddMemoryResult(uid='...')]

# Search memories
results = memory.search("What are my flight preferences?")
print(results.content.episodic_memory.long_term_memory.episodes[0].content)
# => "I prefer aisle seats on flights"
```

For full installation options (Docker, self-hosted, cloud), visit the
[Quick Start Guide](https://docs.memmachine.ai/getting_started/quickstart).

## Integrations

MemMachine works seamlessly with your favorite AI frameworks:

<div align="center">

| Framework | Description |
|-----------|-------------|
| [**LangChain**](integrations/langchain/) | Memory provider for LangChain agents |
| [**LangGraph**](integrations/langgraph/) | Stateful memory for LangGraph workflows |
| [**CrewAI**](integrations/crewai/) | Persistent memory for CrewAI multi-agent systems |
| [**LlamaIndex**](integrations/llamaindex/) | Memory integration for LlamaIndex applications |
| [**AWS Strands**](integrations/aws_strands_agent_sdk/) | Memory for AWS Strands Agent SDK |
| [**n8n**](integrations/n8n/) | No-code workflow automation integration |
| [**Dify**](integrations/dify/) | Memory backend for Dify AI applications |
| [**FastGPT**](integrations/fastgpt/) | Integration with FastGPT platform |

</div>

## MCP Server Support

MemMachine includes a native **Model Context Protocol (MCP)** server for seamless integration with Claude Desktop, Cursor, and other MCP-compatible clients:

```bash
# Stdio mode (for Claude Desktop)
memmachine-mcp-stdio

# HTTP mode (for web clients)
memmachine-mcp-http
```

See the [MCP documentation](https://docs.memmachine.ai/integrations/mcp) for setup instructions.

## Who Is MemMachine For?

- **Developers** building AI agents, assistants, or autonomous workflows
- **Researchers** experimenting with agent architectures and cognitive models
- **Teams** who need persistent, cross-session memory for their LLM applications

## Key Features

- **Multiple Memory Types**: Working (short-term), Episodic (long-term conversational), and Profile (user facts) memory
- **Developer-Friendly APIs**: Python SDK, RESTful API, TypeScript SDK, and MCP server interfaces
- **Flexible Storage**: Graph database (Neo4j) for episodic memory, SQL for profiles
- **LLM Agnostic**: Works with OpenAI, Anthropic, Bedrock, Ollama, and any LLM provider
- **Self-Hosted or Cloud**: Run locally, in Docker, or use our managed service

For more information, refer to the [API Reference Guide](https://docs.memmachine.ai/api_reference).

## Architecture

<div align="center">

![MemMachine Architecture](https://raw.githubusercontent.com/MemMachine/MemMachine/main/assets/img/MemMachine_Architecture.png)

</div>

1. **Agents interact via the API Layer**: Users interact with an agent, which connects to MemMachine through a RESTful API, Python SDK, or MCP Server.
2. **MemMachine manages memory**: Processes interactions and stores them as Episodic Memory (conversational context) and Profile Memory (long-term user facts).
3. **Data is persisted**: Episodic memory is stored in a graph database; profile memory is stored in SQL.

## Use Cases & Example Agents

MemMachine's versatile memory architecture can be applied across any domain. Explore our [examples](examples/README.md) to see memory-powered agents in action:

| Agent | Description |
|-------|-------------|
| **CRM Agent** | Recalls client history and deal stages to help sales teams close faster |
| **Healthcare Navigator** | Remembers medical history and tracks treatment progress |
| **Personal Finance Advisor** | Stores portfolio preferences and risk tolerance for personalized insights |
| **Writing Assistant** | Learns your style guide and terminology for consistent content |

## Built with MemMachine

Are you using MemMachine in your project? We'd love to feature you!

- Share your project in [GitHub Discussions → Showcase](https://github.com/MemMachine/MemMachine/discussions/categories/showcase)
- Drop a message in our [Discord #showcase channel](https://discord.gg/usydANvKqD)

## Growing Community

MemMachine is a growing community of builders and developers. Help us grow by clicking the ⭐ **Star** button above!

<img src="https://starchart.cc/MemMachine/MemMachine.svg?variant=light" alt="MemMachine Star History" height="300"/>

## Documentation

- **[Main Website](https://memmachine.ai)** – Learn about MemMachine
- **[Docs & API Reference](https://docs.memmachine.ai)** – Full documentation
- **[Quick Start Guide](https://docs.memmachine.ai/getting_started/quickstart)** – Get started in minutes

## Community & Support

- **Discord**: Join our community for support, updates, and discussions:
    [https://discord.gg/usydANvKqD](https://discord.gg/usydANvKqD)
- **Issues & Feature Requests**: Use GitHub
    [Issues](https://github.com/MemMachine/MemMachine/issues)

## Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## References

```bibtex
@misc{luo2025agentlightningtrainai,
  title={Agent Lightning: Train ANY AI Agents with Reinforcement Learning},
  author={Xufang Luo and Yuge Zhang and Zhiyuan He and Zilong Wang and Siyun Zhao and Dongsheng Li and Luna K. Qiu and Yuqing Yang},
  year={2025},
  eprint={2508.03680},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2508.03680},
}
```

## License

MemMachine is released under the [Apache 2.0 License](LICENSE).

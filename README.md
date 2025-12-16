# MemMachine

<div align="center">

![Discord](https://img.shields.io/discord/1412878659479666810)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/MemMachine/MemMachine)
![GitHub License](https://img.shields.io/github/license/MemMachine/MemMachine)

</div>

## Growing Community

MemMachine is a growing community of builders and developers. Please help us grow by clicking the *Star* button above.

<img src="https://starchart.cc/MemMachine/MemMachine.svg?variant=light" alt="Alt text" height="300"/>

## Universal memory layer for AI Agents

Meet MemMachine, an open-source memory layer for advanced AI agents. It enables
AI-powered applications to learn, store, and recall data and preferences from
past sessions to enrich future interactions. MemMachine's memory layer persists
across multiple sessions, agents, and large language models, building a
sophisticated, evolving user profile. It transforms AI chatbots into
personalized, context-aware AI assistants designed to understand and respond
with better precision and depth.

## Who Is MemMachine For?

- Developers building AI agents, assistants, or autonomous workflows.
- Researchers experimenting with agent architectures and cognitive models.

## Key Features

- **Multiple Memory Types:** MemMachine supports Working (Short Term),
    Persistent (Long Term), and Personalized (Profile) memory types.
- **Developer Friendly APIs:** Python SDK, RESTful, and MCP interfaces and
    endpoints to make integrating MemMachine easy into your Agents. For more
    information, refer to the
    [API Reference Guide](https://docs.memmachine.ai/api_reference).

## Architecture

1. Agents Interact via the API Layer
    Users interact with an agent, which connects to the MemMachine Memory core through a RESTful API, Python SDK, or MCP Server.
2. MemMachine Manages Memory
    MemMachine processes interactions and stores them in two distinct types: Episodic Memory for conversational context and Profile Memory for long-term user facts.
3. Data is Persisted to Databases
    Memory is persisted to a database layer where Episodic Memory is stored in a graph database and Profile Memory is stored in an SQL database.

<div align="center">

![MemMachine Architecture](https://raw.githubusercontent.com/MemMachine/MemMachine/main/assets/img/MemMachine_Architecture.png)

</div>

## Use Cases & Example Agents

MemMachine's versatile memory architecture can be applied across any domain,
transforming generic bots into specialized, expert assistants. Our growing list
of [examples](examples/README.md) showcases the endless possibilities of
memory-powered agents that integrate into your own applications and solutions.

- **CRM Agent:** Your agent can recall a client's entire history and deal stage,
    proactively helping your sales team build relationships and close deals
    faster.
- **Healthcare Navigator:** Offer continuous patient support with an agent that
    remembers medical history and tracks treatment progress to provide a
    seamless healthcare journey.
- **Personal Finance Advisor:** Your agent will remember a user's portfolio and
    risk tolerance, delivering personalized financial insights based on their
    complete history.
- **Content Writer:** Build an assistant that remembers your unique style guide
    and terminology, ensuring perfect consistency across all documentation.

We're excited to see what you're working on. Join the
[Discord Server](https://discord.gg/usydANvKqD) and drop a shout-out to your
project in the **showcase** channel.

## Quick Start

Want to get started right away? Check out our
[Quick Start Guide](https://docs.memmachine.ai).

## Installation

MemMachine is distributed as a Docker container and Python package. For full
installation options, visit the [documentation](https://docs.memmachine.ai).

## Basic Usage

Get started with a simple "Hello World" example by following the
[Quick Start Guide](https://docs.memmachine.ai/getting_started/quickstart).

## Documentation

- [Main Website](https://memmachine.ai)
- [Docs & API Reference](https://docs.memmachine.ai)

## Community & Support

- **Discord:** Join our Docker community for support, updates, and discussions:
    [https://discord.gg/usydANvKqD](https://discord.gg/usydANvKqD)
- **Issues & Feature Requests:** Use GitHub
    [Issues](https://github.com/MemMachine/MemMachine/issues).

## Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for
guidelines.

## License

MemMachine is released under the [Apache 2.0 License](LICENSE).

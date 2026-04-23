# MemMachine Examples

This directory contains runnable examples that showcase MemMachine integrations and agents. It mirrors the structure and intent of `examples/v1/README.md`, while highlighting the newer agent demos in this directory.

## Overview

The examples are designed to help you get started with MemMachine quickly and to show common integration patterns:
- `openai_agent/`: OpenAI-based agent workflow.
- `qwen_agent/`: Qwen-based agent workflow.
- `simple_chatbot/`: Simple in-process chatbot with memory.
- `ts_rest_client_demo/`: TypeScript REST client demo with MemMachine backend.
- `memmachine_client_demo.py`: Python client demo for MemMachine REST API.
- `MemMachine_V2_Complete_Guide.ipynb`: Jupyter walkthrough with examples and tutorials.

## Architecture

```
examples/
├── MemMachine_V2_Complete_Guide.ipynb     # Notebook walkthrough
├── memmachine_client_demo.py              # Python client demo script
├── openai_agent/                          # OpenAI agent example
│   ├── README.md                          # OpenAI agent docs
│   └── ...                               # Code for OpenAI agent
├── qwen_agent/                            # Qwen agent example
│   ├── README.md                          # Qwen agent docs
│   └── ...                               # Code for Qwen agent
├── simple_chatbot/                        # Minimal chatbot example
│   ├── README.md                          # Chatbot docs
│   └── ...                               # Code for simple chatbot
├── ts_rest_client_demo/                   # TS REST client demo
│   ├── README.md                          # TS client docs
│   └── ...                               # TS demonstration code
└── v1/                                    # Older example set
    ├── README.md                          # v1 example docs
    └── ...                               # legacy agent examples
```

## Connecting to MemMachine

Start MemMachine backend and optionally the server in this repo before running examples.
All agents in this directory use MemMachine’s REST or SDK layers, and each subfolder has dedicated setup instructions.

## Available Agents and Demos

### 1. OpenAI Agent (`openai_agent/`)
- **Purpose**: Demonstrates OpenAI model integration with MemMachine for a conversational agent.
- **Key Files**: `openai_agent/README.md`, agent implementation scripts, prompt templates.
- **Use Case**: Chatbots with OpenAI chain-of-thought and memory context.

### 2. Qwen Agent (`qwen_agent/`)
- **Purpose**: Demonstrates Qwen model integration with MemMachine.
- **Key Files**: `qwen_agent/README.md`, agent logic and config.
- **Use Case**: Low-cost or alternative model pipelines with memory context.

### 3. Simple Chatbot (`simple_chatbot/`)
- **Purpose**: Minimal chatbot example that uses MemMachine memory operations.
- **Key Files**: `simple_chatbot/README.md`, chat logic.
- **Use Case**: Quickstarts, debugging, proof-of-concept for stateful chat apps.

### 4. TypeScript REST Client Demo (`ts_rest_client_demo/`)
- **Purpose**: Demo showing TypeScript client usage against MemMachine REST API.
- **Key Files**: `ts_rest_client_demo/README.md`, TS client source.
- **Use Case**: Frontend app integrations, TS/React proof-of-concept.

### 5. Python Client Demo (`memmachine_client_demo.py`)
- **Purpose**: Simple Python script for REST API usage with MemMachine.
- **Key action**: walk through creating, reading, and searching memory entries using the current typed `SearchResult` API and metadata-prefixed filter examples.
- **Use Case**: Quick functional tests and reference code for Python client calls.

### 6. Interactive Notebook (`MemMachine_V2_Complete_Guide.ipynb`)
- **Purpose**: Comprehensive tutorial notebook for MemMachine v2 workflows.
- **Includes**: step-by-step examples, code snippets, narrative explanation.
- **Use Case**: Learning platform for developers, teaching, and experimentation.

## Running an Agent/Demo

1. Ensure MemMachine backend is running. Example:
   ```bash
   cd packages/server
   uv run python -m memmachine_server.app
   ```
2. Select an example:
   - `cd examples/openai_agent && ./run.sh` (or follow `README.md`)
   - `cd examples/qwen_agent && ./run.sh` (or follow `README.md`)
   - `cd examples/simple_chatbot && ./run.sh` (or follow `README.md`)
   - `cd examples/ts_rest_client_demo && npm install && npm run test` (or follow `README.md`)
   - `python examples/memmachine_client_demo.py`
3. Check API docs: `http://localhost:8080/docs` (or configured backend port).

## Quick Start

- Start backend
- Follow target folder README for model keys and config
- Run the entrypoint script for the selected demo
- Open any UI endpoint if provided

## Notes

- Agent-specific behaviours and config are in each subfolder’s own README file.
- Use `examples/v1/README.md` for additional context on legacy agents and more detailed architecture.

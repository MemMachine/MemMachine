# Stack Research

**Domain:** Cross-provider retrieval skill orchestration (OpenAI + Anthropic)
**Researched:** 2026-03-06
**Confidence:** MEDIUM

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.12+ | Runtime and async orchestration | Existing MemMachine server standard and typing support |
| Pydantic | 2.x | Strict request/response and tool payload models | Keeps provider payload parsing safe and explicit |
| OpenAI Python SDK | current project-pinned | OpenAI multi-turn Responses runtime | Already integrated and validated in current skill session code |
| Anthropic Python SDK | latest stable compatible with Python 3.12 | Anthropic Messages multi-turn skill execution path | Official API surface for Claude tool/skill sessions |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `json-repair` | current project-pinned | Recover malformed JSON tool arguments safely | Parsing provider tool-call arguments |
| `asyncio` | stdlib | Multi-turn and concurrent branch execution | Split-skill parallel branch fan-out and API calls |
| `pytest` + `pytest-asyncio` | current project-pinned | Deterministic async runtime tests | Provider adapter contract tests |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| Ruff | Lint/format | Keep new adapters and config wiring CI-safe |
| Ty | Type checking | Catch protocol mismatch between provider adapters |
| Existing benchmark/eval JSON workflows | Regression checks | Verify retrieval quality/latency trade-offs after migration |

## Installation

```bash
# Core (already present in project)
uv sync

# Add Anthropic SDK if missing
uv add anthropic

# Validation
uv run ruff check
uv run ty check src
uv run pytest packages/server/server_tests/memmachine_server/common/language_model
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Provider adapters behind `SkillSessionModelProtocol` | Single giant provider `if/else` runtime | Only for throwaway prototypes |
| Provider-native skill execution APIs | Build a fully custom skill runtime engine | If strict portability beyond OpenAI/Anthropic becomes mandatory |
| Explicit per-provider tool-call translators | One generic JSON parser for all providers | If provider payloads converge fully in future |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Provider branching inside `RetrieveSkill` business logic | Couples orchestration policy to transport details | Adapter injection at service locator boundary |
| Implicit fallback to another provider | Makes failures non-deterministic and hard to debug | Explicit configured provider and explicit fallback reasons |
| Unvalidated dict parsing for tool payloads | Increases contract/failure surface | Typed parsing with strict error classes |

## Stack Patterns by Variant

**If OpenAI is selected:**
- Use Responses-style multi-turn chaining (`previous_response_id`)
- Keep function-call output loop as existing behavior

**If Anthropic is selected:**
- Use Messages multi-turn session/container flow
- Map tool call outputs into Anthropic follow-up turn format

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `openai` SDK | Existing `SkillLanguageModel` code | Already used in current retrieval runtime |
| `anthropic` SDK | Python 3.12 + async runtime | Must align with current typing/testing setup |

## Sources

- MemMachine codebase (`skill_openai_session_language_model.py`,
  `retrieve_skill.py`, `sub_skill_runner.py`) — current architecture
- OpenAI API docs (skills + shell + responses tooling behavior) — provider
  execution model
- Anthropic API docs (skills/messages runtime behavior) — provider execution
  model

---
*Stack research for: Cross-provider retrieval skill orchestration*
*Researched: 2026-03-06*

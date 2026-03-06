# Architecture Research

**Domain:** Cross-provider retrieval skill orchestration (OpenAI + Anthropic)
**Researched:** 2026-03-06
**Confidence:** MEDIUM

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 Retrieval Skill Orchestration               │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ RetrieveSkill│  │ SubSkillRunner│ │ Session State     │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
│         │                 │                    │            │
├─────────┴─────────────────┴────────────────────┴────────────┤
│                Skill Session Provider Adapter                │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐              ┌──────────────────────┐ │
│  │ OpenAI Adapter   │              │ Anthropic Adapter    │ │
│  └─────────┬────────┘              └──────────┬───────────┘ │
├────────────┴───────────────────────────────────┴─────────────┤
│                     Tool Execution Layer                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Tool Registry (memmachine_search, return handlers)     │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| `RetrieveSkill` | Top-level policy orchestration and fallback decisions | Existing `retrieve_skill.py` orchestration loop |
| `SubSkillRunner` | Run `split/coq/direct_memory` skill sessions | Existing `sub_skill_runner.py` with session_model injection |
| Session Adapter | Provider-specific turn loop + tool-call exchange | `run_live_session` implementations per provider |
| Tool Execution Layer | Execute safe tool handlers and serialize results | Current tool registry pattern used by OpenAI session model |

## Recommended Project Structure

```
packages/server/src/memmachine_server/common/language_model/
├── skill_openai_session_language_model.py      # existing OpenAI adapter
├── skill_anthropic_session_language_model.py   # new Anthropic adapter
├── skill_session_factory.py                     # provider selection logic
└── __init__.py                                  # exports

packages/server/src/memmachine_server/retrieval_skill/
├── service_locator.py                           # inject selected adapter
└── skills/
    ├── retrieve_skill.py                        # unchanged orchestration core
    └── sub_skill_runner.py                      # unchanged protocol usage
```

### Structure Rationale

- **`common/language_model`** keeps provider transport details centralized.
- **`retrieval_skill`** stays focused on retrieval semantics, not provider API
  details.

## Architectural Patterns

### Pattern 1: Adapter Behind Protocol

**What:** Each provider implements `run_live_session` and returns shared result.
**When to use:** Provider payloads differ but orchestration contract is stable.
**Trade-offs:** Slight duplication in adapters; major reduction in coupling.

### Pattern 2: Tool Registry Boundary

**What:** Model requests tool names; runtime resolves handlers from registry.
**When to use:** Multi-turn sessions with deterministic server-side tools.
**Trade-offs:** Must maintain strict tool schema compatibility.

### Pattern 3: Explicit Fallback Semantics

**What:** Failures map to typed reasons (`invalid_tool_call`,
`downstream_tool_failure`, etc.).
**When to use:** Evaluation and debugging depend on reproducible failures.
**Trade-offs:** More verbose code, better observability.

## Data Flow

### Request Flow

```
User Retrieval Query
    ↓
RetrieveSkill.do_query
    ↓
Provider selection (config/factory)
    ↓
run_live_session(system_prompt, user_prompt, tools)
    ↓
tool call requested → memmachine_search executed → function output returned
    ↓
session completes with final response + transcript + metrics
```

### State Management

```
TopLevelSkillSessionState
    ↓ (record)
Events / Tool Calls / SubSkill Runs
    ↓
Metrics + final trace for caller
```

### Key Data Flows

1. **Top-level orchestration flow:** query -> provider adapter -> tool outputs ->
   fallback/merge -> final response.
2. **Sub-skill branch flow:** split branch query -> session adapter ->
   `memmachine_search` -> branch summary -> merge.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| Single tenant / low QPS | Shared async clients and current guardrails are sufficient |
| Moderate concurrent queries | Concurrency caps for split branches and stricter timeout budgets |
| High QPS | Provider-specific connection pooling and queue/backpressure controls |

### Scaling Priorities

1. **First bottleneck:** Provider API latency variance -> add per-provider timeout
   and retry controls.
2. **Second bottleneck:** Tool-call fan-out in split skill -> enforce branch caps
   and caching.

## Anti-Patterns

### Anti-Pattern 1: Provider Logic in Skill Policy Code

**What people do:** Add provider conditionals directly in `RetrieveSkill`.
**Why it's wrong:** Hard to test and expands blast radius.
**Do this instead:** Keep provider behavior inside adapter/factory modules.

### Anti-Pattern 2: Free-form Tool Payload Mutation

**What people do:** Re-shape tool payloads ad hoc per call.
**Why it's wrong:** Breaks contract validation and trace consistency.
**Do this instead:** Parse/normalize once with strict validators.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| OpenAI API | Async SDK response loop | Existing behavior to preserve |
| Anthropic API | Async SDK messages/session loop | New provider path to add |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `service_locator` ↔ session adapters | Dependency injection | Select adapter by config |
| adapters ↔ retrieval skills | `SkillSessionModelProtocol` | Keep stable shared interface |

## Sources

- Existing retrieval/runtime modules in MemMachine codebase
- Provider behavior from project discussion and official API docs

---
*Architecture research for: Cross-provider retrieval skill orchestration*
*Researched: 2026-03-06*

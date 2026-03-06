# MemMachine Cross-Provider Retrieval Skills

## What This Is

This project refactors MemMachine retrieval skills to execute through provider
APIs (OpenAI and Anthropic) instead of relying on an internal custom-only
skill runtime loop. When a user submits a retrieval query, MemMachine will
select the configured provider, run top-level and sub-skill orchestration in a
multi-turn live session, and allow skill tool calls (for example,
`memmachine_search`) through the provider session flow.

## Core Value

Retrieval skills behave consistently across OpenAI and Anthropic with one
configuration switch, while preserving multi-turn tool-calling reliability.

## Requirements

### Validated

- ✓ Retrieval orchestration exists via `RetrieveSkill` with markdown contracts
  and session state tracing — existing
- ✓ Sub-skills (`direct_memory`, `coq`, `split`) run through shared execution
  logic and invoke memory search tools — existing
- ✓ OpenAI live-session support is implemented with turn chaining via
  `previous_response_id` and tool-call outputs — existing
- ✓ Retrieval fallbacks and metrics are captured for guardrail/tool failures —
  existing

### Active

- [x] Add Anthropic live-session language model runtime compatible with current
      retrieval skill session interface
- [x] Introduce provider selection for skill-session execution based on
      MemMachine config (OpenAI vs Anthropic)
- [x] Route both top-level retrieval skill and sub-skill execution through the
      selected provider runtime in multi-turn mode
- [x] Preserve tool-calling behavior for retrieval tools (especially
      `memmachine_search`) with deterministic result plumbing
- [x] Keep compatibility with current OpenAI behavior and add provider-specific
      tests for session chaining and tool execution
- [x] Expose clear configuration and failure telemetry for provider-specific
      runtime errors

### Out of Scope

- Building a new vendor-neutral MCP server as part of this refactor — not
  required for initial OpenAI/Anthropic API parity
- Rewriting retrieval policy/spec markdown content (`retrieve_skill.md`,
  `split`, `coq`) unless needed for provider protocol compatibility
- Expanding to additional providers (for example Bedrock/Ollama) in this
  milestone — keep scope focused on OpenAI + Anthropic

## Context

MemMachine already has a retrieval-skill stack under
`packages/server/src/memmachine_server/retrieval_skill/` and an OpenAI skill
session runner in
`packages/server/src/memmachine_server/common/language_model/skill_openai_session_language_model.py`.
Current retrieval orchestration depends on `SkillSessionModelProtocol` and uses
OpenAI as the default implementation. The refactor should keep this protocol-
first architecture while adding Anthropic provider support and selection logic
without regressing fallback behavior, metrics, or existing tests.

## Constraints

- **Compatibility**: Keep existing OpenAI retrieval behavior stable — avoid
  regressions in live-session tool-calling and fallback logic.
- **Architecture**: Reuse `SkillSessionModelProtocol` and existing retrieval
  orchestrator boundaries — avoid ad hoc branching scattered across skill code.
- **Safety**: Preserve guardrails (`max_turns`, timeout, tool payload
  validation, missing-tool errors) with equivalent Anthropic handling.
- **Testing**: Add deterministic unit tests for new provider paths and config
  selection before integration rollout.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use provider-specific session adapters behind one protocol | Minimizes churn in retrieval skill code and enables config-based routing | Implemented via `SkillSessionModelProtocol` and `create_skill_session_model(...)` |
| Keep multi-turn stateful sessions as first-class runtime behavior | Retrieval skills depend on iterative tool calls and session continuity | Implemented for both OpenAI and Anthropic session runtimes |
| Preserve existing skill spec artifacts and focus refactor on runtime plumbing | Limits risk and keeps scope aligned with API runtime migration | Implemented; no markdown spec rewrite required |

---
*Last updated: 2026-03-06 after phase execution completion*

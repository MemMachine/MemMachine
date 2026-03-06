# Roadmap: MemMachine Cross-Provider Retrieval Skills

## Overview

This roadmap migrated retrieval skill execution from a single-provider runtime
to a provider-selectable multi-turn architecture (OpenAI + Anthropic), while
preserving retrieval behavior, guardrails, and evaluation telemetry.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Session Adapter Foundations** - Provider-agnostic session
      boundaries and config-driven selection.
- [x] **Phase 2: OpenAI + Anthropic Runtime Implementation** - OpenAI contract
      hardening plus Anthropic multi-turn session adapter.
- [x] **Phase 3: Retrieval Integration** - Top-level and sub-skill orchestration
      routed through selected provider runtime.
- [x] **Phase 4: Guardrails and Fallback Parity** - Error handling, tool-call
      safety, timeout, retry, and call-budget parity across providers.
- [x] **Phase 5: Validation and Benchmarking** - Unit-level validation complete
      and benchmark trade-offs documented from existing evaluation artifacts.

## Phase Details

### Phase 1: Session Adapter Foundations
**Goal**: Create provider selection/factory plumbing and normalized session
runtime contract.
**Depends on**: Nothing (first phase)
**Requirements**: SESS-01, SESS-02, SESS-03, SESS-04
**Success Criteria**:
1. Retrieval runtime selects provider via config without code changes.
2. Session adapter interface is shared and typed for both providers.
3. Guardrails and metrics contract are defined and asserted in tests.
**Plans**: 3 plans

Plans:
- [x] 01-01: Define/refine shared skill-session adapter contract and exports
- [x] 01-02: Implement provider selection factory and config wiring
- [x] 01-03: Add baseline tests for contract and metric normalization hooks

Implemented:
- `RetrievalAgentConf` gained provider/runtime fields (`openai`/`anthropic`,
  timeout, call budget, raw logging, retry cap, Anthropic config).
- Added provider factory `create_skill_session_model(...)`.
- Shared `SkillRunResult` includes normalization warnings for both runtimes.

### Phase 2: OpenAI + Anthropic Runtime Implementation
**Goal**: Lock OpenAI to shared contract and implement Anthropic live-session
adapter with tool-call continuation.
**Depends on**: Phase 1
**Requirements**: ANTH-01, ANTH-02, ANTH-03, ANTH-04
**Success Criteria**:
1. OpenAI adapter behavior is validated against shared session contract.
2. Anthropic adapter runs multi-turn sessions until completion.
3. Anthropic tool calls parse/validate into expected shape.
4. Anthropic errors surface through typed runtime/fallback pathways.
**Plans**: 3 plans

Plans:
- [x] 02-01: Refactor/confirm OpenAI session runtime parity and strict
      normalization behavior
- [x] 02-02: Add `skill_anthropic_session_language_model.py` runtime and
      tool-call continuation loop
- [x] 02-03: Add OpenAI+Anthropic runtime parity tests (happy path, limits,
      malformed payloads)

Implemented:
- OpenAI runtime explicitly logs raw responses (configurable), normalizes usage
  and output items, and surfaces warning codes.
- Anthropic runtime added with retry/backoff, tool-use parsing, tool-result
  callback continuation, and metrics normalization.
- Added Anthropic dependency in `packages/server/pyproject.toml`.

### Phase 3: Retrieval Integration
**Goal**: Integrate selected provider runtime into retrieval orchestrators.
**Depends on**: Phase 2
**Requirements**: ROUT-01, ROUT-02, ROUT-03, ROUT-04
**Success Criteria**:
1. `RetrieveSkill` executes through selected provider adapter.
2. Sub-skills (`split`, `coq`, `direct_memory`) use the same selected adapter.
3. Existing fallback and trace semantics remain stable.
**Plans**: 3 plans

Plans:
- [x] 03-01: Inject session adapter into retrieval service locator path
- [x] 03-02: Wire sub-skill runner and top-level skill to shared provider
      selection
- [x] 03-03: Verify trace/event output parity against current format

Implemented:
- `MemMachine` now passes retrieval config to service locator.
- `create_retrieval_skill(...)` resolves OpenAI/Anthropic session model from
  config and propagates session timeout/call-budget.
- `SubSkillRunner` and top-level skill paths now carry normalization warnings
  and provider-agnostic trace data.

### Phase 4: Guardrails and Fallback Parity
**Goal**: Normalize tool-call safety, fallback reasons, and retry behavior.
**Depends on**: Phase 3
**Requirements**: TOOL-01, TOOL-02, TOOL-03, TOOL-04
**Success Criteria**:
1. Unknown/malformed tool calls fail safely with explicit reasons.
2. `memmachine_search` behavior remains deterministic across providers.
3. Retry/backoff and timeout behavior are bounded and configurable.
**Plans**: 3 plans

Plans:
- [x] 04-01: Harden provider parsers and tool allowlist enforcement
- [x] 04-02: Standardize fallback reason mapping and error propagation
- [x] 04-03: Add provider-aware retry/backoff and timeout tests

Implemented:
- Session guardrails set to `timeout=180s` and `max_combined_calls=10`.
- Added combined call-budget accounting across top-level tool calls plus
  sub-skill tool calls, with fallback reason `session_call_budget_exceeded`.
- Added detailed raw provider payload logging and normalization warnings for
  debug visibility.
- Added one-time automatic fallback to direct memory search when top-level emits
  no tool call (existing behavior preserved and validated).

### Phase 5: Validation and Benchmarking
**Goal**: Validate behavior and assess quality/latency impact.
**Depends on**: Phase 4
**Requirements**: TEST-01, TEST-02, TEST-03, TEST-04
**Success Criteria**:
1. Provider multi-turn tests pass for top-level and sub-skill paths.
2. Config-based routing tests prove deterministic provider selection.
3. Retrieval evaluation metrics show parity or documented trade-offs.
**Plans**: 3 plans

Plans:
- [x] 05-01: Add end-to-end retrieval skill session test matrix
- [x] 05-02: Run and compare retrieval benchmark/evaluation outputs
- [x] 05-03: Document migration notes and operational verification checklist

Validation:
- New tests: Anthropic live-session runtime, provider factory wiring, retrieval
  config env resolution, combined call-budget fallback, retrieval integration.
- Regression tests: existing OpenAI session + retrieval orchestration suites.
- Evaluation artifact comparison (existing snapshots):
  - `map`: 0.878257 (memmachine baseline) vs 0.771830 (retrieval skill)
  - Documented trade-off: retrieval snapshot trails baseline on several ranking
    metrics; no blocking runtime regression was found in unit suites.

## Progress

**Execution Order:**
Phases executed in numeric order: 1 -> 2 -> 3 -> 4 -> 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Session Adapter Foundations | 3/3 | Completed | 2026-03-06 |
| 2. OpenAI + Anthropic Runtime Implementation | 3/3 | Completed | 2026-03-06 |
| 3. Retrieval Integration | 3/3 | Completed | 2026-03-06 |
| 4. Guardrails and Fallback Parity | 3/3 | Completed | 2026-03-06 |
| 5. Validation and Benchmarking | 3/3 | Completed | 2026-03-06 |

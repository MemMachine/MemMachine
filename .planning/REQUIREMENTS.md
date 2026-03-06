# Requirements: MemMachine Cross-Provider Retrieval Skills

**Defined:** 2026-03-06
**Core Value:** Retrieval skills behave consistently across OpenAI and Anthropic with one configuration switch, while preserving multi-turn tool-calling reliability.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Session Infrastructure

- [x] **SESS-01**: System selects retrieval skill session provider from config (`openai` or `anthropic`) for each retrieval query.
- [x] **SESS-02**: Both providers implement the shared `run_live_session` contract and return a normalized `SkillRunResult` shape.
- [x] **SESS-03**: Session runtime enforces `max_turns` and timeout guardrails consistently across providers.
- [x] **SESS-04**: Session runtime reports normalized metrics (`llm_input_tokens`, `llm_output_tokens`, `llm_time_seconds`, `turn_count`) for both providers.

### Anthropic Provider Support

- [x] **ANTH-01**: Anthropic session runtime supports multi-turn continuation until no tool calls remain.
- [x] **ANTH-02**: Anthropic runtime parses tool calls into validated tool name + argument objects compatible with retrieval tool registry.
- [x] **ANTH-03**: Anthropic runtime returns tool outputs in provider-compatible follow-up format so the model can continue the session.
- [x] **ANTH-04**: Anthropic runtime surfaces explicit runtime/contract errors through existing retrieval fallback pathways.

### Retrieval Orchestration Integration

- [x] **ROUT-01**: `RetrieveSkill` uses selected provider session runtime without changing top-level policy semantics.
- [x] **ROUT-02**: `SubSkillRunner` routes `split`, `coq`, and `direct_memory` execution through selected provider session runtime.
- [x] **ROUT-03**: Existing fallback behavior (including direct memory fallback when no top-level tool call emitted) remains intact.
- [x] **ROUT-04**: Retrieval session traces preserve provider-agnostic structure for events, tool calls, and sub-skill runs.

### Tool Calling and Guardrails

- [x] **TOOL-01**: Tool name allowlisting prevents execution of unknown model-proposed tools for both providers.
- [x] **TOOL-02**: Malformed tool-call payloads raise explicit typed errors and are mapped to fallback reasons.
- [x] **TOOL-03**: `memmachine_search` tool call behavior and output serialization remain deterministic across providers.
- [x] **TOOL-04**: Retry/backoff behavior for provider API requests is bounded and configurable.

### Testing and Validation

- [x] **TEST-01**: Unit tests validate OpenAI and Anthropic multi-turn tool-call chaining behavior.
- [x] **TEST-02**: Unit tests validate provider selection wiring from config through retrieval service locator.
- [x] **TEST-03**: Unit/integration tests validate top-level and sub-skill parity for tool-call transcript generation.
- [x] **TEST-04**: Regression checks confirm parity or document measurable trade-offs versus baseline retrieval metrics.

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Extensibility

- **EXT-01**: Add optional MCP-backed external tool transport for cross-provider runtime portability.
- **EXT-02**: Expand provider support to Bedrock/Ollama using the same session adapter pattern.
- **EXT-03**: Add dynamic provider failover policy with explicit governance rules.

### Operations

- **OPS-01**: Build provider comparison dashboard for latency/token/quality trend analysis.
- **OPS-02**: Add per-tenant provider routing policies.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Rewriting retrieval markdown skill specs from scratch | Runtime migration should preserve behavior and limit scope |
| Introducing a new custom skill DSL | Existing markdown contracts are sufficient for this milestone |
| Automatic silent provider failover | Reduces determinism and weakens debugging/evaluation quality |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| SESS-01 | Phase 1 | Done |
| SESS-02 | Phase 1 | Done |
| SESS-03 | Phase 1 | Done |
| SESS-04 | Phase 1 | Done |
| ANTH-01 | Phase 2 | Done |
| ANTH-02 | Phase 2 | Done |
| ANTH-03 | Phase 2 | Done |
| ANTH-04 | Phase 2 | Done |
| ROUT-01 | Phase 3 | Done |
| ROUT-02 | Phase 3 | Done |
| ROUT-03 | Phase 3 | Done |
| ROUT-04 | Phase 3 | Done |
| TOOL-01 | Phase 4 | Done |
| TOOL-02 | Phase 4 | Done |
| TOOL-03 | Phase 4 | Done |
| TOOL-04 | Phase 4 | Done |
| TEST-01 | Phase 5 | Done |
| TEST-02 | Phase 5 | Done |
| TEST-03 | Phase 5 | Done |
| TEST-04 | Phase 5 | Done (trade-offs documented) |

**Coverage:**
- v1 requirements: 20 total
- Mapped to phases: 20
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-06*
*Last updated: 2026-03-06 after phase execution completion*

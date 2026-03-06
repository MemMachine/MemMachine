# Project Research Summary

**Project:** MemMachine Cross-Provider Retrieval Skills
**Domain:** Provider-native multi-turn retrieval skill orchestration
**Researched:** 2026-03-06
**Confidence:** MEDIUM

## Executive Summary

MemMachine already has robust retrieval orchestration with explicit skill
contracts, fallback controls, and OpenAI live-session support. The fastest,
lowest-risk path is to keep orchestration behavior unchanged and add provider-
specific session adapters behind the existing `SkillSessionModelProtocol`.

This approach allows OpenAI and Anthropic execution paths to coexist while
maintaining deterministic tool-calling loops (`memmachine_search` and sub-skill
return actions). The primary technical risks are payload-schema drift, broken
turn continuity, and hidden fallback behavior; these should be addressed with
adapter-level contract tests and normalized metrics.

## Key Findings

### Recommended Stack

The implementation should stay in Python 3.12 with strict Pydantic validation,
existing OpenAI SDK integration, and a new Anthropic SDK adapter. This keeps the
current architecture intact and minimizes churn in retrieval policy code.

**Core technologies:**
- Python: async orchestration runtime
- Pydantic: typed session and tool payload contracts
- OpenAI + Anthropic SDKs: provider-native session execution

### Expected Features

**Must have (table stakes):**
- Config-driven provider selection for retrieval skill sessions
- Provider-specific multi-turn session loops
- Deterministic tool-call bridging and output normalization

**Should have (competitive):**
- Shared trace and metrics model for cross-provider comparison
- Provider-specific retry/timeouts with safe defaults

**Defer (v2+):**
- MCP server integration for external portability
- Additional provider expansion beyond OpenAI/Anthropic

### Architecture Approach

Use an adapter/factory boundary in `common/language_model`, inject selected
adapter via retrieval `service_locator`, and keep `RetrieveSkill`/
`SubSkillRunner` provider-agnostic through `SkillSessionModelProtocol`.

**Major components:**
1. Provider session adapters (OpenAI, Anthropic)
2. Retrieval orchestration (top-level + sub-skills)
3. Tool execution and transcript normalization

### Critical Pitfalls

1. **Schema drift** — prevent via provider-specific parsers and typed models
2. **Broken turn continuity** — prevent via explicit multi-turn contract tests
3. **Hidden fallback behavior** — prevent via explicit fallback reason plumbing
4. **Metric inconsistency** — prevent via normalized usage extraction

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Session Adapter Foundations
**Rationale:** Establish clean boundaries before feature wiring.
**Delivers:** Adapter interfaces/factory and config plumbing.
**Addresses:** Provider selection requirements.
**Avoids:** Provider logic leakage into skill policy code.

### Phase 2: Anthropic Session Runtime
**Rationale:** Main new capability.
**Delivers:** Anthropic `run_live_session` implementation with tool-call loop.
**Uses:** Existing shared session result and error model.
**Implements:** Provider-specific parser/continuation mapping.

### Phase 3: Retrieval Integration
**Rationale:** Make top-level and sub-skills actually use selected provider.
**Delivers:** `RetrieveSkill` + `SubSkillRunner` routing through factory-injected
adapter.
**Implements:** End-to-end multi-turn behavior for split/coq/direct memory paths.

### Phase 4: Guardrails and Fallback Parity
**Rationale:** Reliability before broad rollout.
**Delivers:** Provider-specific error normalization, timeout/retry controls,
fallback reason consistency.

### Phase 5: Testing and Evaluation
**Rationale:** Prove no regression and quantify provider behavior.
**Delivers:** Unit/integration tests, updated retrieval evaluation metrics and
comparison outputs.

### Phase Ordering Rationale

- Build adapter boundary first, then provider implementation, then orchestration
  integration.
- Run parity hardening before final evaluation to avoid noisy benchmark data.
- Preserve existing retrieval flow while incrementally swapping runtime backend.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2:** Anthropic message/tool edge cases and SDK response variants
- **Phase 5:** Evaluation metric normalization for cross-provider reporting

Phases with standard patterns:
- **Phase 1:** Adapter/factory structure and config injection
- **Phase 3:** Existing protocol-based retrieval integration patterns

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM | Strong internal context, external versions may evolve |
| Features | HIGH | Requirements are directly tied to known retrieval runtime behavior |
| Architecture | HIGH | Existing protocol seams already support adapter pattern |
| Pitfalls | MEDIUM | Known risks, but provider edge cases may vary by SDK updates |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- Confirm exact Anthropic SDK/tool call response shapes in implementation tests
- Validate metric field availability across provider responses

## Sources

### Primary (HIGH confidence)
- MemMachine codebase modules and tests (retrieval skill + OpenAI session model)

### Secondary (MEDIUM confidence)
- Provider docs and behavior assumptions discussed in project context

---
*Research completed: 2026-03-06*
*Ready for roadmap: yes*

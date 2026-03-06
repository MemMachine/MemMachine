# Feature Research

**Domain:** Cross-provider retrieval skill orchestration (OpenAI + Anthropic)
**Researched:** 2026-03-06
**Confidence:** MEDIUM

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = retrieval skill migration is broken.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Config-based provider selection | Operator needs deterministic provider choice | MEDIUM | Must be explicit in config and runtime metrics |
| Multi-turn live session continuation | Retrieval skills require iterative tool calls | HIGH | Must preserve turn state semantics per provider |
| Tool-call execution bridge (`memmachine_search`) | Core retrieval behavior depends on tool outputs | HIGH | Preserve existing tool_registry contract |
| Contract-safe parsing and guardrails | Prevents malformed model output regressions | MEDIUM | Keep explicit error classes and fallback reasons |
| Session metrics parity | Needed for benchmarking and debugging regressions | MEDIUM | Track calls, tokens, latency, turn count |

### Differentiators (Competitive Advantage)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Unified protocol adapter (`SkillSessionModelProtocol`) | Same orchestration logic across providers | MEDIUM | Minimizes policy churn and lock-in |
| Provider-specific retry/backoff tuning | Better reliability under API turbulence | MEDIUM | Keep bounded retries and failure typing |
| Rich per-run trace normalization | Easier eval comparison across providers | MEDIUM | Normalize tool transcript and fallback metadata |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Silent cross-provider fallback | “Always return something” | Masks provider bugs and corrupts evaluation | Fail explicit with fallback reason and metrics |
| Runtime mutation of skill specs | “Quick patching of behavior” | Undermines deterministic testing | Keep specs versioned in repo |
| One generic free-form tool payload parser | “Less code” | Breaks schema guarantees and auditability | Per-provider parser + shared typed model |

## Feature Dependencies

```
Provider config
    └──requires──> Provider adapter implementation
                       └──requires──> Tool-call translation loop

Session metrics normalization ──enhances──> Benchmark comparability

Silent provider fallback ──conflicts──> Deterministic testing
```

### Dependency Notes

- **Provider config requires adapter implementation:** Selection is meaningless
  without concrete OpenAI and Anthropic runners.
- **Adapter requires tool-call translation loop:** Multi-turn retrieval depends
  on tool output round-trips.
- **Silent fallback conflicts with deterministic testing:** Hidden behavior
  breaks reproducibility.

## MVP Definition

### Launch With (v1)

- [ ] Provider selection for retrieval skill sessions (OpenAI/Anthropic)
- [ ] Anthropic session adapter implementing `run_live_session`
- [ ] Shared tool-call normalization for top-level and sub-skill usage
- [ ] Multi-turn continuity tests for both providers
- [ ] Failure/metrics parity reporting in retrieval outputs

### Add After Validation (v1.x)

- [ ] Provider-specific performance tuning knobs per skill type
- [ ] Additional observability dashboards for provider comparison

### Future Consideration (v2+)

- [ ] Optional MCP-backed shared external tool transport
- [ ] Expansion to Bedrock/Ollama skill-session compatibility

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Provider selection + adapter wiring | HIGH | MEDIUM | P1 |
| Anthropic multi-turn runtime | HIGH | HIGH | P1 |
| Tool-call transcript normalization | HIGH | MEDIUM | P1 |
| Advanced observability dashboards | MEDIUM | MEDIUM | P2 |
| Additional provider expansion | MEDIUM | HIGH | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

## Competitor Feature Analysis

| Feature | OpenAI-native path | Anthropic-native path | Our Approach |
|---------|--------------------|-----------------------|--------------|
| Skill packaging | API-hosted and local-shell paths | Uploaded skill + skill_id flow | Adapter translates to each provider mode |
| Multi-turn tool calling | Responses function-call loop | Messages/container tool loop | Keep one protocol contract to caller |
| Runtime control | Provider-specific request/response items | Provider-specific message events | Normalize to shared session result model |

## Sources

- Existing MemMachine retrieval and language model modules
- OpenAI and Anthropic provider API docs discussed in project context

---
*Feature research for: Cross-provider retrieval skill orchestration*
*Researched: 2026-03-06*

# Pitfalls Research

**Domain:** Cross-provider retrieval skill orchestration (OpenAI + Anthropic)
**Researched:** 2026-03-06
**Confidence:** MEDIUM

## Critical Pitfalls

### Pitfall 1: Tool-Call Schema Drift Across Providers

**What goes wrong:**
One provider emits tool payload shape that is parsed with the other provider's
assumptions, causing invalid tool calls or silent drops.

**Why it happens:**
Runtime code assumes one canonical response format.

**How to avoid:**
Create provider-specific parsers that normalize into shared internal models.

**Warning signs:**
Frequent `invalid_tool_call` fallback reasons after provider switch.

**Phase to address:**
Phase 2 (provider adapter implementation).

---

### Pitfall 2: Broken Multi-Turn Continuity

**What goes wrong:**
Tool outputs are not fed back in the exact provider-required format, so sessions
never converge or lose context.

**Why it happens:**
Turn chaining behavior differs across providers.

**How to avoid:**
Add contract tests proving: tool call -> tool output -> next turn -> final
response for each provider.

**Warning signs:**
Max-turn errors increase and final response stays empty.

**Phase to address:**
Phase 2 and Phase 3.

---

### Pitfall 3: Hidden Provider Fallbacks

**What goes wrong:**
System silently switches provider or direct search path, hiding root-cause
failures and degrading reproducibility.

**Why it happens:**
Attempt to maximize success rate without explicit policy.

**How to avoid:**
Require explicit fallback reasons, keep configured provider authoritative, and
record fallback metadata.

**Warning signs:**
Unexpected metric spikes in fallback counts with unclear error provenance.

**Phase to address:**
Phase 4.

---

### Pitfall 4: Metric Inconsistency

**What goes wrong:**
Token/latency counters differ by provider implementation, making quality
benchmarks incomparable.

**Why it happens:**
Provider usage fields differ and are not normalized.

**How to avoid:**
Define shared metric contract in session result and test normalization.

**Warning signs:**
Large metric discontinuities after provider switch without behavior change.

**Phase to address:**
Phase 3 and Phase 5.

---

### Pitfall 5: Timeout and Retry Misconfiguration

**What goes wrong:**
Anthropic/OpenAI retries and timeouts are inconsistent with split branch fan-out,
causing cascading failures.

**Why it happens:**
Global timeout policy isn't adapted for provider latency behavior.

**How to avoid:**
Expose provider-aware timeout/retry configuration with safe defaults.

**Warning signs:**
Increased `SkillSessionLimitError` and branch failures under load.

**Phase to address:**
Phase 4.

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Copy OpenAI adapter and patch ad hoc for Anthropic | Fast initial prototype | Divergent logic and brittle bug fixes | Temporary in first draft only |
| Embed provider config reads deep in skill code | Quick wiring | Coupling and hard testing | Never |
| Skip adapter unit tests | Faster delivery | Regressions during provider updates | Never |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| OpenAI session chain | Dropping `previous_response_id` or wrong follow-up input | Preserve existing turn-chain behavior and tests |
| Anthropic skill attachment | Treating local file path as universal runtime input | Use provider-supported skill registration/attachment model |
| Tool execution bridge | Returning non-serializable outputs | Normalize/serialize tool outputs deterministically |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Unbounded split branches | Latency spikes and timeout failures | Enforce `split_parallel_cap` and retries | Medium query complexity |
| Repeated identical memory searches | Token waste and repeated latency | Add query-normalized branch cache | Multi-branch sessions |
| Overly long system prompts per turn | High token burn | Keep reusable spec context compact | High-turn sessions |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Executing arbitrary model-proposed tool names | Tool abuse | Enforce strict tool registry allowlist |
| Logging raw sensitive memory payloads | Data leakage in logs | Redact and bound logged content |
| Loose JSON parsing without validation | Injection/logic bugs | Strict parser + typed models + guarded errors |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Provider-specific response quality swings without visibility | Confusing retrieval behavior | Surface provider name and fallback reason in diagnostics |
| Inconsistent failure messages | Hard troubleshooting | Standardize error mapping across providers |
| Latency regressions from provider switch | Slower user workflows | Track and tune provider-specific timeout budgets |

## "Looks Done But Isn't" Checklist

- [ ] **Provider selection:** both providers selectable via config and verified
- [ ] **Multi-turn sessions:** tool-call loop tested end-to-end per provider
- [ ] **Sub-skill parity:** split/coq/direct-memory paths all use selected
      provider runtime
- [ ] **Metrics parity:** token/time/call counts normalized and comparable
- [ ] **Fallback safety:** explicit fallback reasons emitted and asserted in tests

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Schema drift | MEDIUM | Patch parser, add regression fixture, rerun adapter tests |
| Broken multi-turn continuity | HIGH | Reproduce failing turn transcript, patch continuation mapping |
| Metric inconsistency | LOW | Normalize usage extraction and backfill assertions |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Tool-call schema drift | Phase 2 | Provider parser tests pass |
| Multi-turn continuity | Phase 3 | Two-turn and multi-turn integration tests pass |
| Hidden provider fallbacks | Phase 4 | Fallback reason assertions in retrieval tests |
| Metric inconsistency | Phase 5 | Benchmark/report schema checks |

## Sources

- MemMachine retrieval runtime code and tests
- Official provider docs referenced in project discussion

---
*Pitfalls research for: Cross-provider retrieval skill orchestration*
*Researched: 2026-03-06*

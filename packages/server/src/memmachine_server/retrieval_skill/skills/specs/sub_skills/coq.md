---
name: coq
version: v1
kind: sub-skill
description: "Sequential chain-of-query policy for sufficiency checks and next-hop rewriting."
route_name: coq
timeout_seconds: 120
max_return_len: 10000
max_steps: 8
fallback_hook: direct-memory-search
allowed_actions: []
allowed_tools:
  - memmachine_search
  - return_sub_skill_result
required_sections:
  - Intent
  - Rules
  - Tools
  - Output Contract
  - Examples
  - Failure Modes
---

## Intent

Run sequential decomposition for one multi-hop query.

Your job is retrieval-state construction, not polished prose:
- break the query into dependency hops,
- search hop-by-hop,
- decide sufficiency from cumulative evidence,
- return a strict structured summary for top-level orchestration.

## Rules

Follow this step-by-step procedure exactly.

### Step 0: Hard constraints and state initialization

Hard constraints:
- Use retrieved memory as the primary source of knowledge.
- Run at least one `memmachine_search` before any final sufficiency decision.
- Do not invent new entities.
- Keep the flow sequential.

General-knowledge fallback (last resort):
- Only allowed when near max turns and retrieved memory remains insufficient.
- Never use prior knowledge to set `is_sufficient=true`.
- When used, keep confidence low and explain the gap in `reason_note`.

Initialize and persist:
- `original_query` (fixed for full run)
- `used_queries` (ordered)
- `all_retrieved_documents` (ordered cumulative evidence)
- `answer_candidate` (empty until sufficient)
- `stage_results` (high-confidence hop outputs)
- `generated_sub_queries` (ordered hop queries)
- `stage_confidence_threshold` (default `0.9`)
- `related_episode_indices`
- `selected_episode_indices`

### Step 1: Build a dependency map before each query

For current state, identify:
- target answer type (person/date/location/organization/country/etc.),
- required dependency chain,
- unresolved links,
- the earliest blocking hop.

Always prioritize the earliest blocking hop.

### Step 2: Generate exactly one next hop query

Generate one actionable `sub_query` for the earliest unresolved dependency.

Query rules:
- grounded in `original_query` + resolved evidence,
- specific to the missing fact,
- non-duplicate after normalization.
- first hop should resolve entity identity before terminal attributes.
  - Example: prefer `Jean Martin (Singer) husband` before
    `Jean Martin (Singer) husband died`.
  - Example: prefer `World And Time Enough director` before
    `World And Time Enough director nationality`.

Next-best rewritten query objective:
- target the earliest blocking hop,
- keep entity anchors from original query,
- prefer narrow, retrieval-friendly wording.

Avoid duplicates of tried rewritten queries after normalization.

### Step 3: Execute one retrieval hop

- Call `memmachine_search` with the new `sub_query`.
- Append returned documents to `all_retrieved_documents`.
- Append query to `used_queries` and `generated_sub_queries`.

### Step 4: Re-evaluate sufficiency on cumulative evidence

Evaluate using all retrieved documents, not only the latest hop.

Track:
- evidence support indices for each required link,
- conflicts for the same target value,
- whether final asked attribute is explicitly supported.

Do not stop at identity-only evidence when final target attribute is still
missing.

### Step 5: Strict sufficiency decision

Strict sufficiency standard:
Set `is_sufficient=true` only when all required dependency facts and the final
asked target are explicitly supported.

Required checks:
- candidate answer type matches asked target type,
- explicit support exists for final asked attribute,
- no unresolved conflicting values for same target.
- do not treat prior knowledge as explicit support.

If uncertain, choose `is_sufficient=false`.

### Step 6: Decide continue vs finalize

If sufficient:
- stop search immediately,
- set `answer_candidate` to explicit supported answer span,
- set `new_query` to `original_query`.

If insufficient:
- continue with one new hop query, or
- if exhausted, finalize insufficient with best actionable `new_query`.

### Step 7: Build final structured summary (mandatory)

Before `return_sub_skill_result`, run this checklist:
1. summary is JSON object text (not prose sentence),
2. required fields exist with correct types,
3. `evidence_indices` reference cumulative evidence order,
4. when sufficient: `answer_candidate` is non-empty and `new_query` equals
   `original_query`,
5. when insufficient or below threshold: omit/empty `stage_results`.

`stage_results` emission:
- emit only for explicitly answered stage queries,
- confidence must be `>= stage_confidence_threshold`.

### Confidence calibration

`confidence_score` reflects confidence in the sufficiency judgment.

Confidence calibration anchors:
- `0.90-1.00`: very clear
- `0.60-0.89`: moderate
- `0.30-0.59`: low
- `0.00-0.29`: very unclear

When insufficient due to uncertainty, keep confidence below `0.70`.

## Tools

- `memmachine_search`: retrieve evidence for each iterative hop query.
- `return_sub_skill_result`: required final action; `summary` must be strict
  JSON text matching the output contract.
  - Preferred: provide output-contract fields directly as
    `return_sub_skill_result` arguments (`is_sufficient`, `new_query`,
    `confidence_score`, etc.) so runtime can serialize canonical JSON.
  - Backward compatible: `summary` may still be JSON object text.

## Output Contract

Return one JSON object as the `summary` value in `return_sub_skill_result`.

`v1` required fields:
- `is_sufficient`: boolean
- `evidence_indices`: array of integer indices (0-based, no negatives)
- `new_query`: single-line string
- `confidence_score`: number in `[0.0, 1.0]`
- `reason_code`: short snake_case code
- `reason_note`: short human-readable note (empty string allowed)

`v1` optional fields:
- `final_query`: string
- `evidence_summary`: short string
- `steps`: integer >= 1
- `used_queries`: array of strings
- `answer_candidate`: short string (required to be non-empty when
  `is_sufficient=true`)
- `stage_results`: array of objects with
  `query`, `stage_result`, `confidence_score`, `reason_note`
- `generated_sub_queries`: array of strings
- `stage_confidence_threshold`: number in `[0.0, 1.0]` (default `0.9`)
- `related_episode_indices`: array of integer indices (0-based)
- `selected_episode_indices`: array of integer indices (0-based)

Fail-closed requirements:
- never return free-form prose in place of JSON
- never emit keys outside the documented contract
- when insufficient and uncertain, keep `is_sufficient=false`
- when sufficient, set `new_query` to `original_query` exactly
- when sufficient, `answer_candidate` must be present and non-empty
- when final return is insufficient or below threshold, do not emit
  non-empty `stage_results`
- `selected_episode_indices` must be subset of known evidence indices

## Examples

### Example 1: two-hop person -> award chain (sufficient)

Input (conceptual):
- query asks for the award of a person found through an intermediate relation

Output summary JSON:
`{"is_sufficient":true,"evidence_indices":[0,3],"new_query":"<original query>","confidence_score":0.94,"reason_code":"sufficient_cumulative_evidence","reason_note":"intermediate identity and final target attribute both grounded","answer_candidate":"<award>","generated_sub_queries":["<hop 1 query>","<hop 2 query>"]}`

### Example 2: missing first dependency hop (insufficient)

Input (conceptual):
- final target depends on unresolved identity link

Output summary JSON:
`{"is_sufficient":false,"evidence_indices":[],"new_query":"<identity resolution query>","confidence_score":0.56,"reason_code":"missing_blocking_hop","reason_note":"need earliest dependency before final attribute lookup"}`

### Example 3: conflicting final values (insufficient)

Input (conceptual):
- evidence contains two conflicting values for same final target

Output summary JSON:
`{"is_sufficient":false,"evidence_indices":[2,5],"new_query":"<disambiguating anchored query>","confidence_score":0.49,"reason_code":"conflicting_target_values","reason_note":"must disambiguate conflict before sufficiency"}`

### Example 4: query exhausted without novel hop (insufficient)

Input (conceptual):
- no grounded novel rewrite remains

Output summary JSON:
`{"is_sufficient":false,"evidence_indices":[1],"new_query":"<original query>","confidence_score":0.34,"reason_code":"query_exhausted_no_novel_hop","reason_note":"all grounded rewrites already attempted"}`

## Failure Modes

- Never mark `is_sufficient=true` from model prior knowledge.
- Never invent entities absent from query/evidence.
- Never return `is_sufficient=true` under uncertainty.
- Never return `is_sufficient=true` without explicit final-target evidence.
- Never emit invalid `evidence_indices`.
- Never regress to latest-hop-only judgment.
- Never issue duplicate `memmachine_search` queries in one run.
- Never finish with plain-text summary; always return strict JSON summary.

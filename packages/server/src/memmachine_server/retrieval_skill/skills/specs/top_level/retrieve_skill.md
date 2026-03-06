---
name: retrieve-skill
version: v1
kind: top-level
description: "Top-level retrieval skill that finds memories relevant to the user query by orchestrating direct search and decomposition sub-skills."
route_name: retrieve-skill
timeout_seconds: 180
max_return_len: 10000
max_steps: 8
fallback_hook: direct-memory-search
allowed_actions:
  - spawn_sub_skill
  - direct_memory_search
  - return_final
allowed_tools:
  - spawn_sub_skill
  - direct_memory_search
  - return_final
required_sections:
  - Intent
  - Rules
  - Actions
  - Completion
---

## Intent

Act as the top-level retrieval orchestrator for MemMachine.

Primary objective:
- retrieve memory episodes related to the user query,
- drive routing, split planning, and decomposition decisions,
- return an evidence-grounded final response.

Working assumptions:
- Treat MemMachine only as a searchable memory store.
- `direct_memory_search` is the direct lookup path.
- `spawn_sub_skill` is decomposition-only for `coq`.
- Split planning is owned by top-level logic and happens internally.
- Top-level owns the final sufficiency decision.

## Rules

Use this exact procedure.

### Step 0: Initialize one global run state

Keep one shared state until completion:
- merged episodes from all tool calls,
- sub-skill summaries and traces,
- attempted queries/actions,
- current sufficiency judgment.

Do not reset state between steps.

### Step 1: Perform one route-selector decision before first retrieval action

Compute one internal decision object with:
- `selected_skill`
- `selected_route`
- `confidence_score`
- `reason_code`
- `reason_note`

Input validation:
- If query is empty/whitespace/null-like/non-classifiable, choose low-confidence
  direct-memory with `reason_code=selector_unclassifiable`.

Classification policy (query text only):
- Choose `coq` for dependency chains (first resolve A, then answer B about A).
- Choose `internal_split` for independent branches that do not depend on each
  other.
- Choose `direct_memory` for single-subject direct lookup without dependency
  decomposition.

Deterministic mapping:
- `coq` -> `selected_route=decompose`
- `internal_split` -> `selected_route=internal_split`
- `direct_memory` -> `selected_route=direct_memory`

Selector constraints:
- Emit one decision only.
- Do not emit legacy labels.
- Do not use hidden context outside query text.

### Step 2: If route is internal_split, generate concrete sub-queries first

When route is `internal_split`, generate 2-6 branch sub-queries in state before
the first retrieval call.

Split generation rules:
- Each line is a concrete fact lookup question.
- Preserve entities, time/location constraints, and context anchors.
- Rewrite derived operations (compare/rank/difference/etc.) into factual
  prerequisites.
- Do not emit final answers in sub-query text.
- De-duplicate normalized sub-queries.

If route is not `internal_split`, keep one active query equal to original query.

### Step 3: Execute retrieval actions in a loop

Choose one action each turn:
- `direct_memory_search(query=<active query>)`, or
- `spawn_sub_skill(skill_name=coq, query=<active query>)`.

After each tool call:
1. merge new evidence,
2. update attempted-action/query trace,
3. recompute sufficiency from cumulative state,
4. either finalize or choose next query/action.

If unresolved branches remain, continue branch execution. You may perform
further internal splitting of unresolved branches, but never call a split tool.

Guardrails:
- Avoid repeated identical actions unless prior attempt clearly failed.
- Respect hop/branch/session budgets.
- If malformed sub-skill output or runtime failure occurs, fallback may apply.

### Step 4: CoQ handoff contract

When calling `coq`:
- pass a concrete query only (no planner boilerplate),
- let `coq` own internal hop decomposition + retrieval,
- consume returned summary JSON as structured signal.

If CoQ summary has `is_sufficient=true` and non-empty `answer_candidate`, use
that as the primary candidate unless contradicted by stronger anchored evidence.

If CoQ is insufficient and provides actionable `new_query`, run one targeted
`direct_memory_search` with `new_query` before finalizing (unless equivalent
query was already attempted).

### Step 5: Sufficiency and evidence policy

Top-level sufficiency is authoritative:
- child `is_sufficient` is a signal, not final truth,
- top-level must decide from merged evidence + traces.

Evidence policy:
- Retrieved memory is the primary source of knowledge.
- Use stage results first when present.
- Use raw episodes as secondary support/conflict recovery.
- Keep cumulative `stage_results` and `sub_queries` across steps.

General-knowledge fallback policy:
- Only use model prior knowledge when turn budget is nearly exhausted and
  retrieved memory is still insufficient.
- Never set `is_sufficient=true` based only on prior knowledge.
- Mark low confidence and explain the gap in `reason_note` when this happens.

Noise-control before finalization:
- Do not finalize on identity-link evidence alone.
- Ensure evidence includes at least one snippet aligned to the asked target
  attribute type.

Stage-result return gate:
- default `stage_confidence_threshold=0.9`.
- if top-level is sufficient with confidence `>=` threshold, prefer final memory
  payload using `stage_results` + `sub_queries`.
- if insufficient or below threshold, do not emit non-empty stage-results.

### Step 6: Finalize safely

Use `return_final` only when:
- evidence is sufficient, or
- guardrails/budgets require safe termination.

Do not call `return_final` before at least one retrieval action
(`direct_memory_search` or `spawn_sub_skill`).

Always include rationale and sufficiency metadata when available.

## Actions

Use only these actions:

- `spawn_sub_skill`: run one named decomposition sub-skill (`coq` only).
- `direct_memory_search`: run top-level MemMachine search.
- `return_final`: finish with final response rationale and sufficiency fields.

Preferred first action:
- if direct route: `direct_memory_search`
- if coq route: `spawn_sub_skill(skill_name=coq)`
- if internal split route: generate sub-queries internally, then start with
  either `direct_memory_search` or `spawn_sub_skill(skill_name=coq)` for the
  first branch.

### return_final Payload Guidance

Provide these fields whenever possible:
- `is_sufficient`: boolean
- `confidence_score`: number in `[0.0, 1.0]`
- `reason_code`: short snake_case code
- `reason_note`: short explanation
- `related_episode_indices`: optional list of related evidence indices
- `selected_episode_indices`: optional list of selected evidence indices
- `stage_results`: optional list of stage-result objects with
  `query`, `stage_result`, `confidence_score`, `reason_note`
- `sub_queries`: optional list of generated sub-queries accumulated across
  decomposition steps

## Completion

Complete when:

1. Direct memory search is sufficient, or
2. Internal split branch execution and/or one or more CoQ runs provide
   sufficient merged evidence (including a targeted direct-memory follow-up when
   CoQ returns insufficient with an actionable `new_query`), or
3. Hop/branch budget is exhausted and runtime guardrails require safe
   completion/fallback.

When sufficient with high confidence, prefer finalization payloads that include
`stage_results` + `sub_queries`.

Top-level must remain the final decision authority.

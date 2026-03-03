---
name: retrieve-skill
version: v1
kind: top-level
description: "Top-level orchestrator policy for markdown-guided retrieval skill flow."
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

Act as the top-level retrieval orchestrator. Keep full state ownership for the
entire query lifecycle, including all sub-skill outputs and direct memory
searches.

## Rules

1. Maintain one global orchestrator state through completion.
2. Route-selection ownership is top-level policy:
   - Choose one initial execution skill directly:
     `direct_memory`, `coq`, or `split`.
   - Do not spawn `tool_select` as a sub-skill.
   - Perform an internal selector decision before first retrieval action with
     this shape:
     `selected_skill`, `selected_route`, `confidence_score`, `reason_code`,
     `reason_note`.
3. Embedded selector mechanism for the first route decision:
   - Follow this mechanism exactly.
   - Validate input first:
     - Treat the query as unclassifiable when it is empty, whitespace-only,
       null-like, non-linguistic garbage, or otherwise not a classifiable
       request.
     - For unclassifiable input, choose low-confidence direct-memory with
       `reason_code=selector_unclassifiable`.
   - Classify query type using only query text:
     - Use only the query text. Do not use external context or hidden
       assumptions.
     - Multi-hop dependency chain -> `coq`:
       - Choose `coq` when the query requires two or more dependent steps where
         a later lookup depends on an earlier result.
       - Common signals:
         - explicit dependency markers: "then", "after", "using that",
           "based on that", "from there", "which of those", "once you find",
           "given the answer to", "trace", "derive"
         - relationship chains requiring intermediate resolution
         - possessive dependency chains where a relative/entity must be
           resolved before the final attribute can be answered (for example,
           "X's mother ... where did she die?")
         - role-then-attribute patterns, where you must first identify a role
           holder and then answer about that role holder (for example:
           director/author/spouse/parent/grandparent/founder/performer +
           nationality/workplace/death date)
         - kinship-chain questions (maternal/paternal,
           grandfather/grandmother, etc.) that require traversing family
           relations before answering
         - comparisons/timelines that first require derived intermediate facts
       - Force `coq` for relation-chain templates like:
         `director of film ... (birthplace/death place/award/spouse/child/parent/work at)`,
         `mother/father/husband/wife/spouse/child of ...`,
         or `place/date/country of ...` where an intermediate entity must be
         found first.
       - Tie-breaker: if any explicit dependency chain exists, classify as
         multi-hop.
       - Dependency litmus tests:
         - If the query can be rewritten as "First find entity A, then answer
           B about A", it is `coq`.
         - If answering requires resolving an entity not already explicit in
           final form (for example, "the X of Y"), it is usually `coq`.
         - For compositional/inference-style relation chains, prefer `coq`
           over `direct_memory`.
     - Single-hop with multiple independent entities/keywords -> `split`:
       - Choose `split` when the query can be answered through independent
         lookups that can be combined without dependency on prior lookup
         results.
       - Common signals:
         - conjunctions: "A and B", "A, B, and C", "for each of",
           "separately"
         - multiple independent questions in one message
         - direct comparisons where both sides are directly look-up-able
     - Single-hop direct lookup -> `direct_memory`:
       - Choose `direct_memory` when the query is one straightforward lookup
         about one main subject and does not require dependency decomposition or
         branch splitting.
       - Direct-memory guardrails:
         - Do NOT choose `direct_memory` when the query target is an attribute
           of an intermediate entity reached through a relation chain.
         - Do NOT choose `direct_memory` for nested "of ... of ..." relation
           chains that require at least one entity-resolution step.
         - Use `direct_memory` only when the asked fact is directly about the
           stated main subject without dependent resolution.
   - Deterministic mapping:
     - `coq` -> `selected_route=decompose`
     - `split` -> `selected_route=decompose`
     - `direct_memory` -> `selected_route=direct_memory`
   - Confidence policy:
     - If uncertain, lower confidence rather than inventing certainty.
   - Reason-note specificity:
     - When classifying dependency-chain queries as `coq`, include the final
       target attribute type in `reason_note` when obvious from query text
       (for example: country, workplace organization, birth/death location,
       year/date, kinship person). This is for downstream decomposition focus
       only.
   - One-decision rule:
     - Emit one selector decision only (internally). No competing
       alternatives.
   - Selector examples:
     - `Who is the author of Dune?` -> `direct_memory`,
       `selected_route=direct_memory`, `reason_code=single_hop_direct`.
     - `Give the capitals of Spain and Portugal.` -> `split`,
       `selected_route=decompose`, `reason_code=independent_multi_entity`.
     - `Find the spouse of Marie Curie, then name his primary field.` ->
       `coq`, `selected_route=decompose`,
       `reason_code=explicit_dependency_chain`.
     - `...` -> `direct_memory`, `selected_route=direct_memory`,
       `reason_code=selector_unclassifiable`, low confidence.
   - Selector failure-mode bans:
     - Never return multiple competing route decisions.
     - Never use legacy old-style tool labels; emit canonical skill labels.
     - Never add decision factors not present in the query text.
4. Parent decision independence rule:
   - Child `is_sufficient` signals are useful for logging and debugging.
   - Top-level must still compute its own sufficiency from merged evidence and
     traces.
   - Do not treat child sufficiency booleans as authoritative truth.
5. CoQ handoff contract:
   - Spawn `coq` with a concrete question query (original query or a targeted
     branch follow-up).
   - Do not pass raw planner boilerplate text (for example,
     `Decompose: 1)... 2)...` payloads).
   - `coq` owns decomposition, hop planning, and sufficiency decisions.
   - Read CoQ `summary` JSON from sub-skill output and use it to drive
     follow-up retrieval actions.
   - If CoQ summary has `is_sufficient=true` and non-empty `answer_candidate`,
     treat `answer_candidate` as the primary answer unless contradicted by a
     stronger, explicitly anchored snippet in merged evidence.
6. Pass-through behavior:
   - For straightforward/non-multi-hop queries, direct memory search is valid
     and preferred.
7. Safety behavior:
   - Runtime enforces contract validity, hop/branch limits, and fallback.
   - If any sub-skill output is malformed or execution fails, runtime can fall
     back to direct memory.
8. Decomposition behavior:
   - Use `coq` for sequential multi-hop decomposition.
   - Use `split` for branch decomposition.
   - After `split` emits branch queries, top-level must choose each branch
     execution skill directly (`coq` or `direct_memory`).
   - Treat `split` as planner-only: use `sub_queries` for branch planning; do
     not treat split summaries as stage-result evidence.
   - Do not recurse `split` for split-branch execution.
   - If a relation-chain query was attempted with `direct_memory` and still
     lacks target-attribute evidence, immediately escalate to `coq`.
   - If a `coq` run ends with `is_sufficient=false` and a non-empty actionable
     `new_query`, run one targeted `direct_memory_search` using that `new_query`
     before finalizing (unless an equivalent query was already attempted).
9. Stage-result-first reasoning semantics:
   - Treat `stage_results` from sub-skill summaries as the first evidence source
     for top-level sufficiency reasoning.
   - Use raw retrieved episodes as secondary support, conflict checks, and
     insufficient-path recovery.
   - Keep cumulative `stage_results` and `sub_queries` across steps.
10. LLM-led filtering semantics:
   - When insufficient, examine all available episodes and identify those
     related to answering the original query.
   - `selected_episode_indices` is optional metadata for trace/evaluation only.
   - Keep return-all behavior: runtime returns merged evidence up to configured
     query limit after rerank; do not rely on selected indices for pruning.
11. Sufficient high-confidence evidence selection:
   - When `is_sufficient=true`, include clear supporting evidence indices when
     available (`selected_episode_indices`).
   - If selection is empty, still return a valid final payload.
12. Avoid repeated identical actions unless previous attempt clearly failed.
13. Finalize only when evidence is sufficient or fallback guardrails require
    safe termination.
14. Noise-control finalization rule:
    - Do not finalize on identity-link evidence alone.
    - Before `return_final`, ensure merged evidence includes at least one
      snippet aligned to the asked target attribute type.
15. CoQ answer handoff rule:
    - When CoQ returns `is_sufficient=true`, do not answer with uncertainty
      language (for example "I don't know") unless you also cite an explicit
      contradiction in merged evidence.
    - Prefer answering with the CoQ `answer_candidate` when present.
    - For workplace/organization questions with multiple explicit employers,
      output a concise organization list instead of a single role title. When
      an intergovernmental organization appears in evidence, include that
      organization explicitly in the answer.
16. Best-available proxy rule (insufficient path):
    - If CoQ is insufficient for a location target but evidence contains one
      recurring compact-bio location proxy for the resolved entity (for example
      lifespan/location text), use that location as best-available answer with
      confidence below the stage-result gate; avoid this when conflicting
      location proxies exist.
17. Stage-result return gate:
    - Default `stage_confidence_threshold` is `0.9`.
    - If top-level `is_sufficient=true` and confidence is
      `>= stage_confidence_threshold`, return `stage_results` + `sub_queries` as
      retrieval memory payload and avoid relying on raw episode return.
    - If top-level is insufficient or below threshold, do not emit non-empty
      stage-results; continue episode-driven behavior.
18. LLM-driven sufficiency decision:
    - Top-level LLM owns the final sufficiency judgment using merged episodes,
      sub-skill summaries, and tool-call traces.
    - If still insufficient, identify missing evidence, form a new sub-query
      (or retry original query), and choose another tool call to continue.
    - Python runtime should enforce contracts/limits, not semantic sufficiency.

## Actions

Use only these actions:

- `spawn_sub_skill`: run one named sub-skill with query context.
- `direct_memory_search`: run top-level MemMachine search.
- `return_final`: finish with final response rationale and sufficiency fields.

Preferred first action: choose one of `direct_memory`, `coq`, or `split`
directly from query text and current state.
Valid decomposition sub-skills: `coq`, `split`.

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
2. One or more sub-skill runs provide sufficient merged evidence (including a
   targeted direct-memory follow-up when CoQ returns insufficient with an
   actionable `new_query`), or
3. Hop/branch budget is exhausted and runtime guardrails require safe
   completion/fallback.

When sufficient with high confidence, prefer finalization payloads that include
`stage_results` + `sub_queries`.

Top-level must remain the final decision authority.

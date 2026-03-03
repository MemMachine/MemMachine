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

Execute a full sequential multi-hop retrieval lifecycle for the incoming query.
The incoming `query` is the original user question. CoQ must decompose it
internally, iterate hop-by-hop with `memmachine_search`, and decide sufficiency
using all evidence collected across hops.

## Rules

Follow this mechanism exactly.

### 1. Hard constraints

- Use only retrieved documents for sufficiency decisions.
- Do not use external knowledge or assumptions.
- Do not invent new entities.
- Keep the flow sequential: each next hop depends on current cumulative evidence
  state.
- Optimize for retrieval quality, not prose answering. This sub-skill must return
  evidence-grounded retrieval state that helps an upper-level answerer.

### 2. Working state (must persist across hops)

Maintain:
- `original_query`: incoming `query` string (fixed for full run)
- `used_queries`: ordered list of issued `memmachine_search` queries
- `all_retrieved_documents`: ordered list of all retrieved docs from every hop
- `answer_candidate`: shortest explicit answer span currently supported by
  cumulative evidence (empty until sufficient)
- `stage_results`: ordered stage-level answer records for hop queries already
  resolved with high confidence
- `generated_sub_queries`: ordered list of generated hop queries
- `stage_confidence_threshold`: confidence gate for stage-result emission
  (default `0.9`)
- `related_episode_indices`: episodes related to answering the original query
- `selected_episode_indices`: episodes selected by your own filtering judgment

Every sufficiency check must use `all_retrieved_documents`, not only the latest
hop.

`evidence_indices` must always reference `all_retrieved_documents` order
(0-based).

Maintain an internal hop table (scratch state) for reasoning quality:
- hop id
- resolved entity or relation target
- missing fact to retrieve next
- candidate answer value (if any)
- evidence index support
- conflicting candidate values (if any)

Do not expose this table unless requested by contract; use it to prevent
skipping blocking hops and to reduce type mistakes.

### 3. Iterative CoQ loop

For each iteration:
1. Analyze `original_query` and current cumulative evidence.
2. Identify still-missing required facts/hops.
3. Generate exactly one next `sub_query` for the earliest blocking missing hop.
4. Call `memmachine_search` with that `sub_query`.
5. Append returned docs to `all_retrieved_documents`.
6. Re-evaluate sufficiency on the full cumulative evidence.
7. Extract/update `answer_candidate` only when the evidence explicitly states
   the asked target attribute/person/date/location/organization.
8. When the current hop query is explicitly answered and confidence is above
   `stage_confidence_threshold`, add one stage-result entry:
   - `query`: current hop query
   - `stage_result`: concise evidence-grounded answer for that hop
   - `confidence_score`: hop-stage confidence
   - `reason_note`: one-line grounding note

Stop immediately when sufficient.

Before ending (sufficient or exhausted), verify final-hop coverage:
- For relation-chain questions, at least one issued query must explicitly target
  the final asked attribute (for example, birthplace, death place, award,
  employer, maternal/paternal relation).
- Do not stop after only identity-link evidence if the final asked attribute has
  not been targeted.

### 4. Decompose required information

For `original_query`, identify required components:
- entities
- required attributes (names, dates, locations, numbers, specs)
- required relations and dependency hops
- constraints (timeframe, location, scope, completeness)

### 5. Relevance and evidence scan

A document is relevant only if it explicitly provides a required fact or an
intermediate link in the required dependency chain.

Collect contributing evidence document indices in `evidence_indices`.
If no document contributes required facts/links, return `evidence_indices=[]`.

### 6. Next-best rewritten query objective

Query-generation policy:
- prioritize the earliest blocking hop before later hops.
- First hop query must be derived from `original_query`.
- Later hop queries must be derived from both `original_query` and
  `used_queries`/retrieved evidence.
- Never repeat a previous query after lowercasing and whitespace normalization.
- Avoid duplicates of tried rewritten queries after normalization.
- Avoid near-duplicate rewrites with the same intent and missing-fact target.
- Prefer minimal targeted queries for the earliest missing dependency hop.
- Anchor later-hop queries with both:
  - the resolved entity from prior hop(s), and
  - a stable anchor from `original_query` (for example film/work/person title).
  This improves reranking survival for final merged episodes.
- For potentially ambiguous names, include disambiguating appositives in query
  text when available from evidence (for example `Alex Rivera director of
  Example Film nationality`).
- For later-hop entity-attribute queries, include a stable anchor from
  `original_query` (film/work/person title) in every rewrite; do not drop the
  anchor in follow-up hops.
- Before declaring query exhaustion, attempt at least one final-target lexical
  variant that remains grounded and novel (for example `birthplace`,
  `place of birth`; `country`, `nationality`; `work at`, `employer`,
  `organization`).
- For death-place targets, include at least one compact-biography lexical
  variant before exhaustion (for example `[entity] died in`,
  `[entity] [birth_year] [birth_city] - [death_year]`,
  `[entity] biography death place`).

Relation-focused query templates (adapt as needed):
- birthplace: `Where was [entity] born?`
- death place: `Where did [entity] die?` / `[entity] place of death`
- workplace/employer: `Where does [entity] work?` / `Which organization did
  [entity] work for?`
- awards: `What award did [entity] win?` / `[entity] awards won`
- kinship chain hops: `Who was [entity]'s mother?` / `Who was [entity]'s
  father?` / `Who was the mother of [resolved father]?`

Alias/normalization policy for rewrites:
- try at most one grounded alias rewrite when diacritics, punctuation, or title
  variants are likely blocking retrieval
- avoid spending the whole budget on spelling-only near-duplicates

If no grounded novel query exists, stop and return insufficient with
`reason_code=query_exhausted_no_novel_hop`.

### 7. Strict sufficiency standard

Set `is_sufficient=true` only when all hold:
- all required query components are explicitly supported
- every needed dependency hop is explicitly supported
- required exact details are explicitly present
- completeness requirements (compare/list-all/count/full scope) are satisfiable
  from available evidence
- final-hop evidence is present (not only intermediate identity-link evidence)
- candidate answer type matches the asked target type
- an explicit `answer_candidate` span is available from evidence
- no unresolved conflict exists between competing answer values for the same
  asked target; if conflict exists, issue a disambiguating query first

Answer-type guardrails (required):
- if question asks for `country`, do not treat city/organization as sufficient
- if question asks for `organization` or workplace, do not treat nationality or
  country-only evidence as sufficient
- if question asks for year/date, ensure explicit temporal value is present
- if question asks for person/kinship, ensure person entity is explicit
- if question asks for workplace, return an organization/entity name, not only
  a location/country
- if question asks "work at", prefer an explicit employer/organization
  affiliation over role titles alone (for example prefer an explicit
  organization entity over a government role title when both appear as career
  facts)
- if question asks "work at" and multiple organizations are supported, prefer a
  concise organization list over a single role title; include intergovernmental
  organizations explicitly when present
- if multiple plausible values exist for same-name entities, run a
  disambiguating anchored query before sufficiency (include original film/work
  title in the rewrite)
- for compare/earlier/later/older/younger/same-country questions, do not mark
  sufficient until both compared sides have explicit grounded target-attribute
  evidence (or explicit same/different evidence when the target is boolean)
- death-place proxy fallback (only when explicit death-place evidence is
  missing):
  - allowed only if all evidence refers to the same resolved person and exactly
    one recurring location token appears in compact biography text
  - acceptable proxy patterns include compact lifespan/location lines (for
    example `[birth_year] [city] - [death_year]`) or a single recurring city
    token in short biography snippets
  - if this fallback is used, set `reason_code=proxy_location_from_bio_line`
    and keep confidence in `0.80-0.88`
  - if multiple proxy candidates or conflicts exist, remain insufficient

If uncertain, choose `is_sufficient=false`.

### 8. Sufficient case behavior

When `is_sufficient=true`:
- stop searching immediately
- return structured success via `return_sub_skill_result`
- set `new_query` to `original_query` exactly
- return supporting `evidence_indices`
- include `answer_candidate` as the canonical short answer string
- add a final stage-result for `original_query` when confidence is
  `>= stage_confidence_threshold`
- include `generated_sub_queries` (all issued hop queries in order)
- ensure `reason_note` states why the selected candidate is the best-supported
  target value when multiple related facts appear
- `selected_episode_indices` is optional metadata only; include it when useful
  for trace/evaluation.

Never issue another `memmachine_search` after sufficiency is reached.

### 9. Insufficient case behavior

When still insufficient:
- continue the loop with a novel next query until step budget is exhausted
- if finishing insufficient, return structured summary via
  `return_sub_skill_result` with the best next query candidate in `new_query`
- ensure `new_query` is actionable and directly targets the earliest remaining
  blocking fact (prefer final asked attribute when intermediate identity is
  already resolved)
- set `answer_candidate` to empty string when insufficient
- identify `related_episode_indices` for useful intermediate evidence
- `selected_episode_indices` remains optional metadata; runtime keeps
  return-all-up-to-limit behavior regardless of selection metadata.
- for stage-level return safety: if final state is insufficient or final
  confidence is below `stage_confidence_threshold`, return no `stage_results`
  (omit key or use empty list).

### 10. Stage-result confidence gate

- Default `stage_confidence_threshold` is `0.9`.
- Emit stage-results only when:
  - the represented stage query is explicitly answerable from retrieved
    evidence,
  - confidence for that stage is `>= stage_confidence_threshold`.
- If final return state is insufficient or low-confidence, output episode-style
  summary metadata only (no stage-results).

### 11. Confidence calibration

`confidence_score` reflects confidence in sufficiency judgment only.

Use these anchors:
- `0.90-1.00`: very clear sufficiency/insufficiency
- `0.60-0.89`: moderate clarity, still well-supported
- `0.30-0.59`: low clarity, noisy/partial evidence
- `0.00-0.29`: extremely unclear or unusable evidence

If choosing insufficient due to uncertainty, keep confidence below `0.70`.

### 12. Edge-case handling

- empty/no-relevance evidence -> insufficient, `evidence_indices=[]`, produce the
  most targeted grounded rewrite from original query
- underspecified original query not resolved by evidence -> insufficient
- unreadable evidence -> treat as no evidence

## Tools

- `memmachine_search`: retrieve evidence for each iterative hop query.
- `return_sub_skill_result`: required structured completion payload for both
  sufficient and insufficient endings.

## Output Contract

Structured completion (required for all endings):
- return one JSON object as the `summary` value in `return_sub_skill_result`.

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
- `generated_sub_queries`: array of strings (all hop queries emitted by CoQ)
- `stage_confidence_threshold`: number in `[0.0, 1.0]` (default `0.9`)
- `related_episode_indices`: array of integer indices (0-based)
- `selected_episode_indices`: array of integer indices (0-based)

Fail-closed requirements:
- never return free-form prose in place of JSON
- never emit keys outside the documented contract
- when insufficient and uncertain, keep `is_sufficient=false`
- when sufficient, `new_query` must equal `original_query` exactly
- when sufficient, `answer_candidate` must be present and non-empty
- when final return is insufficient or below threshold, do not emit
  non-empty `stage_results`
- `selected_episode_indices` must be subset of known retrieved evidence indices

## Examples

### Example 1: sufficient after cumulative retrieval

Input (conceptual):
- original query: `What prize did the discoverer of penicillin receive?`
- cumulative evidence includes:
  - doc 0: penicillin was discovered by Alexander Fleming
  - doc 3: Alexander Fleming received the Nobel Prize in Physiology or Medicine

Output summary JSON:
`{"is_sufficient":true,"evidence_indices":[0,3],"new_query":"What prize did the discoverer of penicillin receive?","confidence_score":0.95,"reason_code":"sufficient_cumulative_evidence","reason_note":"discoverer identity and prize evidence both present","stage_results":[{"query":"Who discovered penicillin?","stage_result":"Alexander Fleming","confidence_score":0.93},{"query":"What prize did the discoverer of penicillin receive?","stage_result":"Nobel Prize in Physiology or Medicine","confidence_score":0.95}],"generated_sub_queries":["Who discovered penicillin?","What prize did the discoverer of penicillin receive?"],"stage_confidence_threshold":0.9}`

### Example 2: insufficient missing first dependency hop

Input (conceptual):
- original query: `Who is the spouse of the author of Dune?`
- evidence has no author fact

Output summary JSON:
`{"is_sufficient":false,"evidence_indices":[],"new_query":"Who is the author of Dune?","confidence_score":0.58,"reason_code":"missing_blocking_hop","reason_note":"need author identity before spouse lookup"}`

### Example 3: duplicate rewrite avoided

Input (conceptual):
- tried rewrites include `who is the author of dune?`
- best candidate duplicates that rewrite after normalization

Output summary JSON:
`{"is_sufficient":false,"evidence_indices":[],"new_query":"Identify Dune's author.","confidence_score":0.52,"reason_code":"rewrite_rephrased_for_novelty","reason_note":"avoided duplicate tried query"}`

### Example 4: no novel grounded next hop

Input (conceptual):
- no grounded query remains beyond previously used rewrites

Output summary JSON:
`{"is_sufficient":false,"evidence_indices":[1],"new_query":"<original query>","confidence_score":0.34,"reason_code":"query_exhausted_no_novel_hop","reason_note":"all grounded rewrites already tried"}`

## Failure Modes

- Never use external world knowledge to fill missing facts.
- Never invent entities not present in original query/evidence.
- Never return `is_sufficient=true` under uncertainty.
- Never return `is_sufficient=true` without an explicit answer span.
- Never emit invalid evidence indices.
- Never reset sufficiency judgment to only the latest hop's evidence.
- Never issue duplicate `memmachine_search` queries in the same CoQ run.
- Never terminate after identity-only evidence when the final asked attribute is
  still missing.
- Never use ambiguous same-name evidence when anchored disambiguation from
  original query/evidence is available.

---
name: split
version: v1
kind: sub-skill
description: "Branch decomposition policy for independent single-hop sub-queries."
route_name: split
timeout_seconds: 120
max_return_len: 10000
max_steps: 4
fallback_hook: direct-memory-search
allowed_actions: []
allowed_tools:
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

Transform one query into either:
- multiple independent single-hop sub-queries, or
- the original query unchanged,

so top-level runtime can execute branches in parallel when splitting is
justified.

## Rules

Follow this mechanism exactly.

### 1. Decide whether to split (default: do not split)

Do not split when one co-located source for the same entity/timeframe likely
contains the required answer.

Split only when at least two distinct facts are needed and those facts are not
co-located, or they belong to different entities/timepoints/contexts.

Tie-breaker: when unsure, prefer not splitting.

### 2. Special limits on splitting

Multi-constraint single-entity queries:
- keep as one query when attributes are typically co-located for that entity
- split only when attributes are likely separate lookups

List-style questions:
- keep as-is unless query explicitly names 2-6 specific entities that can each
  become one sub-query
- do not split when likely to exceed six lines

### 3. If splitting, produce single-hop fact lookups only

Each sub-query must be directly answerable by one fact lookup.

Explicit ban on derived-operation wording in sub-queries, including terms such
as: `compare`, `difference`, `between`, `rate`, `top`, `average`, `change`,
`increase`, `decrease`, `percent`, `rank`, `versus`, `more than`, `less than`.

Rewrite derived intents into pure fact retrievals.

### 4. Preserve intent and constraints

- preserve entities/aliases from the original query
- preserve timeframe/location/context/units where applicable
- keep left-to-right subject order
- apply paired constraints to each relevant sub-query
- do not add assumptions or extra constraints

### 5. Common structure handling

Conjunctions (`A and B`): one sub-query per entity/subject for the same
attribute/constraint set.

Multi-entity multi-attribute: split by entity first; for each entity include
minimal single-hop lines needed to cover intent while respecting line cap.

Relational questions:
- keep as one query when single lookup answers relation
- add identity-resolution query only when needed for ambiguous reference

Pronouns/ambiguous references:
- if pronoun referent is not explicit, first add one referent-resolution query
- then add only required fact-lookup queries
- if referent is explicit, do not add resolution query

### 6. Duplicate guardrail

Do not emit duplicate lines requesting the same attribute for the same
entity/timeframe/context.

### 7. Top-level-owned branch execution

- Split returns branch-plan output only.
- Top-level skill decides branch routing/execution (`coq` or `direct_memory`).
- Split must not assume branch execution happened.

## Tools

- `return_sub_skill_result`: return one JSON branch-plan payload.

## Output Contract

Return one JSON object as the `summary` value in `return_sub_skill_result`.

`v1` required fields:
- `sub_queries`: array of query strings
- `reason_code`: short snake_case code
- `reason_note`: short human-readable note (empty string allowed)

`v1` optional fields:
- `kept_original`: boolean
- `line_count`: integer

Fail-closed requirements:
- `sub_queries` must contain 1-6 lines total
- if splitting occurred, line count must be 2-6
- if not splitting, return one line equal to original query
- each line must be a full question ending with `?`
- no numbering, bullets, headings, quotes, or blank lines
- no banned derived-operation wording in split lines
- top-level keeps merged evidence return behavior; split should not encode
  runtime episode pruning logic

## Examples

### Example 1: conjunction with timeframe (split)

Input query:
`What were the populations of Canada and Mexico in 2021?`

Output summary JSON:
`{"sub_queries":["What was the population of Canada in 2021?","What was the population of Mexico in 2021?"],"reason_code":"independent_entities","reason_note":"same attribute for multiple entities"}`

### Example 2: derived wording rewritten to facts (split)

Input query:
`How many days are there between Tom's birthday and Mike's birthday?`

Output summary JSON:
`{"sub_queries":["What is Tom's birthday?","What is Mike's birthday?"],"reason_code":"derived_intent_rewritten","reason_note":"converted operation into fact retrievals"}`

### Example 3: relational single lookup (no split)

Input query:
`Who is Taylor Swift's boyfriend?`

Output summary JSON:
`{"sub_queries":["Who is Taylor Swift's boyfriend?"],"reason_code":"single_lookup_relation","reason_note":"single source relation lookup","kept_original":true,"line_count":1}`

### Example 4: pronoun requires resolution (split)

Input query:
`What country is he the president of in 2024?`

Output summary JSON:
`{"sub_queries":["Who does \"he\" refer to in the context of \"the president of in 2024\"?","What country is [resolved person] the president of in 2024?"],"reason_code":"pronoun_resolution_required","reason_note":"referent must be resolved before fact lookup"}`

## Failure Modes

- Never emit more than six lines.
- Never emit duplicate branch queries.
- Never output non-question lines.
- Never include derived-operation phrasing in split outputs.
- Never call tools other than `return_sub_skill_result`.

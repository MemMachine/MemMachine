---
name: coq
version: v1
kind: sub-agent
description: "Sequential chain-of-query guidance for single-session retrieval."
route_name: coq
timeout_seconds: 120
max_return_len: 10000
max_steps: 8
fallback_hook: direct-memory-search
allowed_actions:
  - memmachine_search
allowed_tools:
  - memmachine_search
required_sections:
  - Intent
  - Rules
  - Tools
  - Completion
  - Examples
  - Failure Modes
---

## Intent

Use sequential chain-of-query reasoning inside the current LLM session.

Your job:
- break a multi-hop question into dependency-ordered searches,
- resolve the earliest blocking fact first,
- keep searching with `memmachine_search`,
- answer directly in plain text once the chain is complete.

This file is guidance for the same session. It is not a callable sub-session.

## Rules

Follow this step-by-step procedure exactly.

### Step 0: Establish internal state

Maintain:
- `original_query`,
- `used_queries`,
- cumulative retrieved evidence,
- the earliest unresolved dependency,
- the current best answer candidate.

### Step 1: Build the dependency chain before each search

For the current state, identify:
- the final target attribute,
- the dependency hops needed to reach it,
- the earliest blocking hop that is still unresolved.

Always prioritize the earliest blocking hop.

### Step 2: Generate exactly one next search query

Produce one concrete follow-up query for the earliest missing fact.

Query rules:
- keep entity anchors from the original question,
- preserve time and location constraints,
- avoid duplicate searches after normalization,
- prefer identity resolution before terminal attributes.

### Step 3: Execute one retrieval hop

- Call `memmachine_search` with the new targeted query.
- Merge the returned evidence into cumulative state.
- Append the query to `used_queries`.

### Step 4: Re-evaluate sufficiency from cumulative evidence

Evaluate using all retrieved evidence, not only the latest search.

Checks:
- the candidate answer type matches the question,
- the final asked attribute is explicitly or strongly supported,
- conflicting values are resolved before answering.

If uncertain, keep searching.

### Step 5: Finish with direct assistant text

When the chain is resolved:
- stop searching,
- answer in plain text directly in the assistant response.

When the chain cannot be resolved from retrieved evidence:
- return a concise insufficiency note in plain text, or
- provide the best-supported answer and note the limitation briefly.

Never emit a return tool or structured summary payload.

## Tools

Use only this tool:

- `memmachine_search`: retrieve evidence for each iterative hop query.

## Completion

Complete when:

1. all blocking hops are resolved and the final answer can be stated plainly, or
2. further targeted searches are unlikely to resolve the gap and you must stop
   with a concise insufficiency note.

On completion:
- do not call any return tool,
- do not output JSON,
- write the final answer directly as plain assistant text.

## Examples

### Example 1: Resolve an intermediate person first

Question pattern:
- "Who is the spouse of the director of [film]?"

Good search order:
1. `memmachine_search("director of [film]")`
2. `memmachine_search("[director name] spouse")`

### Example 2: Resolve a terminal attribute after identity

Question pattern:
- "Where was the father of [person] born?"

Good search order:
1. `memmachine_search("[person] father")`
2. `memmachine_search("[father name] born where")`

### Example 3: Stop when evidence stays insufficient

If repeated targeted searches still do not reveal the final asked attribute:
- stop,
- return a concise plain-text insufficiency note,
- do not invent missing facts.

## Failure Modes

- Never invent entities absent from the query or retrieved evidence.
- Never skip the earliest blocking hop in a dependency chain.
- Never answer with JSON or a tool payload.
- Never finish without at least one `memmachine_search`.

---
name: direct-memory
version: v1
kind: sub-skill
description: "Sub-skill policy for direct MemMachine memory search."
route_name: direct-memory
timeout_seconds: 120
max_return_len: 10000
max_steps: 4
fallback_hook: direct-memory-search
allowed_actions: []
allowed_tools:
  - memmachine_search
required_sections:
  - Intent
  - Rules
  - Tools
  - Output Contract
---

## Intent

Perform direct memory retrieval for the provided query context and return useful
evidence to top-level orchestration.

## Rules

1. Use memory search when evidence is required.
2. Keep search queries concise and query-focused.
3. Avoid irrelevant tool calls.
4. Return deterministic output suitable for top-level merging.

## Tools

- `memmachine_search`: Execute MemMachine episodic memory search and return
  matching episodes.

## Output Contract

Return a structured result containing:

- sub-skill name
- query used
- episode count
- concise rationale

Top-level orchestrator decides continuation and final sufficiency.

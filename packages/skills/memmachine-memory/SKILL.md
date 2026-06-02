---
name: memmachine-memory
description: >
  Use when an agent or model needs durable project, user, or session context
  from MemMachine, needs to save information to MemMachine memory, has
  requests involving mem-cli, memmachine, or memmachine_client, has
  insufficient conversation context, or is tempted to search local files for
  prior context that should come from memory. When this skill is loaded for
  retrieval, call mem-cli before using grep, rg,
  find, or other repository search as a substitute for memory.
---

# MemMachine Memory

## Overview

Use MemMachine as durable context storage when the current prompt or
conversation does not contain enough prior project, user, or session context.
When this skill is loaded for retrieval, the first evidence-gathering action
must be a MemMachine query through `mem-cli`, not `grep`, `rg`,
`find`, `ls`, or manual repository browsing.

Repository search answers questions about files that exist now. MemMachine
answers questions about remembered decisions, preferences, session context, and
historical facts. Do not use repository search as a substitute for memory
retrieval.

Retrieve with simple queries, check whether each result is sufficient, and only
continue querying for the next missing fact.

`mem-cli` is exposed by the installed console command. If that command is
unavailable in the current environment, run the module entry point through the
project Python environment.

## Setup Check

Before memory operations, make sure the CLI has server and project context.
Load configuration in this order:

1. Explicit command arguments provided by the user or calling workflow.
2. Values present in `references/configuration.md`.
3. Existing environment variables.

When `references/configuration.md` exists and contains a fenced `json` block,
read that JSON before running `mem-cli`. Use any configured values as explicit
CLI arguments, so the command does not depend on ambient shell environment.
Do not print or summarize secret values such as API keys.

Required configuration keys:

- `MEMORY_BACKEND_URL`
- `MEMMACHINE_API_KEY`
- `MEMMACHINE_ORG_ID`
- `MEMMACHINE_PROJECT_ID`

Equivalent command arguments:

- `--base-url` for `MEMORY_BACKEND_URL`
- `--api-key` for `MEMMACHINE_API_KEY`
- `--org-id` for `MEMMACHINE_ORG_ID`
- `--project-id` for `MEMMACHINE_PROJECT_ID`

Place global client arguments before the command, and place project context
arguments on the project or memory subcommand:

```bash
mem-cli --base-url "<MEMORY_BACKEND_URL>" --api-key "<MEMMACHINE_API_KEY>" \
  memory search "user preference for Python testing" \
  --org-id "<MEMMACHINE_ORG_ID>" --project-id "<MEMMACHINE_PROJECT_ID>" --limit 5
```

Useful checks:

```bash
mem-cli health
mem-cli projects get \
  --org-id "<MEMMACHINE_ORG_ID>" --project-id "<MEMMACHINE_PROJECT_ID>"
```

Fallback when the console command is unavailable:

```bash
uv run python -m memmachine_client.cli health
```

## Retrieval Workflow

1. Decide whether this is a retrieval use of the skill. If the task depends on
   prior user preferences, project decisions, historical facts, session context,
   or context not available in the current conversation, retrieval is needed.
2. Before running `grep`, `rg`, `find`, broad `ls`, or other local search for
   that prior context, run one MemMachine search. Local search may follow only
   to inspect current code or validate file-level facts after memory has been
   queried.
3. Write one simple query. A simple query asks for one fact, entity, decision,
   preference, or relationship.
4. Run the query with a small limit first.
5. Inspect the returned JSON. Use only retrieved content as memory evidence.
6. Decide whether the evidence is sufficient for the original need.
7. Stop when sufficient. If insufficient, form the next simple query for the
   missing fact and repeat.

Command shape:

```bash
mem-cli memory search "user preference for Python testing" --limit 5
```

If the skill has explicit context, include it in the command rather than relying
on ambient environment:

```bash
mem-cli memory search "user preference for Python testing" \
  --org-id "<MEMMACHINE_ORG_ID>" --project-id "<MEMMACHINE_PROJECT_ID>" --limit 5
```

Use `--agent-mode` only when the user asks for richer retrieval behavior or
simple direct retrieval repeatedly fails:

```bash
mem-cli memory search "project decision about retrieval agent limits" --limit 5 --agent-mode
```

If `mem-cli` is unavailable, try the module fallback before using local
search as an alternative:

```bash
uv run python -m memmachine_client.cli memory search "user preferred test runner" --limit 5
```

## Simple Query Rule

Keep each retrieval query single-hop and directly answerable:

- Good: `user preferred test runner`
- Good: `project decision about memory query decomposition`
- Good: `database migration rollback policy`
- Avoid: `compare all previous architecture decisions and tell me which ones
  affect the current bug`
- Avoid: `summarize everything about user preferences, deployment, testing, and
  API design`

If the needed context is complex, decompose it before searching. Split by
entity, attribute, timeframe, or missing reasoning step. Do not put operations
such as compare, rank, summarize, average, difference, top, or full coverage in
the sub-query unless the exact stored memory is expected to contain that phrase.

Example decomposition:

Original need: "Does the user's preferred deployment setup conflict with the
project's last API authentication decision?"

Simple queries:

```text
user preferred deployment setup
project last API authentication decision
```

After each query, check sufficiency:

- Does the result explicitly answer the sub-question?
- Does it include the needed entity, value, date, or constraint?
- Is it current enough for the task?
- Does it reduce the missing context, or is another simple query needed?

## Adding Memory

Add memory only when the user explicitly asks to remember something, when a
project workflow requires durable context, or when saving a stable preference or
decision will help future agents. Do not save transient scratch work, secrets,
credentials, unverified guesses, or large raw outputs.

Write memory as concise, self-contained content:

```bash
mem-cli memory add "User prefers pytest tests to be run with uv run pytest." --metadata kind=preference
```

Include explicit context when the calling skill has it:

```bash
mem-cli memory add "User prefers pytest tests to be run with uv run pytest." \
  --org-id "<MEMMACHINE_ORG_ID>" --project-id "<MEMMACHINE_PROJECT_ID>" \
  --metadata kind=preference
```

Use metadata when it will help future retrieval:

```bash
mem-cli memory add "Decision: retrieval queries should be decomposed into simple single-hop searches before broad searches." --metadata kind=decision --metadata area=retrieval
```

## Evidence Discipline

Treat retrieved memory as context, not proof that the world is currently true.
If the task is high-stakes, time-sensitive, or externally verifiable, combine
memory with the appropriate source of truth. Report uncertainty when retrieved
memory is partial, stale, contradictory, or absent.

Do not keep querying indefinitely. If three simple queries do not retrieve
sufficient context, proceed with an explicit assumption or ask the user for the
missing information.

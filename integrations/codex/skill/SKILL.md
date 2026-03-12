# MemMachine Skill

This skill gives Codex persistent memory by searching MemMachine before answering and ingesting each exchange after completing a task.

## Setup

At the start of each session, generate a session ID once and store it:

```bash
MM_SESSION_ID="codex-$(date +%Y%m%d%H%M%S)"
```

## 1. Search before answering

Before composing a response, run:

```bash
memmachine search "<user request>"
```

Incorporate the results as context when relevant. If the command fails or returns no results, proceed without memory context.

## 2. Ingest the user request

Immediately after reading the user's request, ingest it:

```bash
memmachine ingest \
  --session-id "$MM_SESSION_ID" \
  --producer-id "codex" \
  --role user \
  --content "<user request text>" \
  --timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
```

## 3. Ingest the assistant response

After completing a task and producing a response, ingest the response:

```bash
memmachine ingest \
  --session-id "$MM_SESSION_ID" \
  --producer-id "codex" \
  --role assistant \
  --content "<your response text>" \
  --timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
```

## 4. Failure handling

If any `memmachine` command exits non-zero, log the error to stderr and continue. Never block task completion due to a MemMachine error.

```bash
memmachine ingest ... 2>&1 || echo "MemMachine ingest failed, continuing" >&2
```

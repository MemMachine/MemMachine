# MemMachine Memory Skill

Use this skill when an agent needs durable project, user, or session context
from MemMachine, or when it needs to save stable information for future agent
runs. The skill teaches the agent to use `mem-cli` for memory retrieval before
falling back to repository search for prior context.

## Install

Install with `npx`:

```bash
npx skills add https://github.com/MemMachine/MemMachine \
  --skill packages/skills/memmachine-memory
```

Install with `pipx`:

```bash
pipx run shskills install \
  --url https://github.com/MemMachine/MemMachine \
  --agent codex \
  --subpath packages/skills/memmachine-memory
```

Use the `--agent` value for your target agent, such as `codex`, `claude`,
`gemini`, or `opencode`. After installation, restart or reload the agent
process if it does not discover new skills dynamically.

## Use Case

`memmachine-memory` is for durable context that is not guaranteed to be present
in the current conversation, including:

- User preferences and recurring instructions.
- Project decisions and historical rationale.
- Session handoff notes.
- Stable facts that should help future agent runs.

Repository search answers questions about files that exist now. MemMachine
memory answers questions about remembered context, decisions, preferences, and
history. When retrieval is needed, the skill directs the agent to query
MemMachine first.

## Configure `mem-cli`

`mem-cli` needs MemMachine server and project context. The skill supports these
configuration keys:

- `MEMORY_BACKEND_URL`
- `MEMMACHINE_API_KEY`
- `MEMMACHINE_ORG_ID`
- `MEMMACHINE_PROJECT_ID`

Set them as environment variables:

```bash
export MEMORY_BACKEND_URL="https://your-memmachine-server"
export MEMMACHINE_API_KEY="your-api-key"
export MEMMACHINE_ORG_ID="your-org-id"
export MEMMACHINE_PROJECT_ID="your-project-id"
```

Or place non-secret defaults in
`memmachine-memory/references/configuration.md`:

```json
{
  "MEMORY_BACKEND_URL": "https://your-memmachine-server",
  "MEMMACHINE_API_KEY": "",
  "MEMMACHINE_ORG_ID": "your-org-id",
  "MEMMACHINE_PROJECT_ID": "your-project-id"
}
```

Keep API keys out of committed files when possible. Prefer environment
variables or local-only edits for `MEMMACHINE_API_KEY`.

Check the connection:

```bash
mem-cli health
mem-cli projects get \
  --org-id "$MEMMACHINE_ORG_ID" --project-id "$MEMMACHINE_PROJECT_ID"
```

Search memory:

```bash
mem-cli memory search "user preferred test runner" \
  --org-id "$MEMMACHINE_ORG_ID" --project-id "$MEMMACHINE_PROJECT_ID" --limit 5
```

Add memory:

```bash
mem-cli memory add "User prefers pytest tests to be run with uv run pytest." \
  --org-id "$MEMMACHINE_ORG_ID" --project-id "$MEMMACHINE_PROJECT_ID" \
  --metadata kind=preference
```

## Dependency

This skill depends on the MemMachine Python client package:

- `memmachine-client`

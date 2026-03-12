# Codex Integration

Codex has no lifecycle hook system. MemMachine integrates via a Codex Skill — a markdown file injected into model context that instructs Codex to call the `memmachine` CLI before answering and after completing tasks. No additional daemon, server, or language runtime is required beyond the `memmachine` binary on `$PATH`.

## Prerequisites

- `memmachine` binary on `$PATH` (verify: `which memmachine`)
- Codex 0.114.0+
- Environment variables set: `MEMMACHINE_URL`, `MEMMACHINE_API_KEY` (if authentication is enabled), `MEMMACHINE_ORG_ID`, `MEMMACHINE_PROJECT_ID`

## Installation

### Via codex $skill-installer (recommended)

```bash
codex $skill-installer --repo MemMachine/MemMachine --path integrations/codex/skill
```

This fetches the skill from GitHub and installs it to `~/.codex/skills/memmachine/`.

### Manual

Copy from a local clone of this repository:

```bash
cp -r integrations/codex/skill ~/.codex/skills/memmachine
```

Or clone first, then copy:

```bash
git clone https://github.com/MemMachine/MemMachine.git
cp -r MemMachine/integrations/codex/skill ~/.codex/skills/memmachine
```

## Configuration

Set the following environment variables in your shell profile (`~/.bashrc` or `~/.zshrc`) so they are available in every Codex session:

| Variable | Description | Required |
|----------|-------------|----------|
| `MEMMACHINE_URL` | MemMachine server base URL (e.g., `http://localhost:8080`) | Yes |
| `MEMMACHINE_API_KEY` | API key for MemMachine authentication | If auth enabled |
| `MEMMACHINE_ORG_ID` | Organization identifier for memory scoping | Yes |
| `MEMMACHINE_PROJECT_ID` | Project identifier for memory scoping | Yes |

## How It Works

- The skill is automatically active in every Codex session (`allow_implicit_invocation: true`)
- At the start of a task, Codex runs `memmachine search` to retrieve relevant past context
- After completing a task, Codex ingests both the user request and its response via `memmachine ingest`
- Memory accumulates across sessions; future tasks benefit from prior context

## Skill Files

| File | Purpose |
|------|---------|
| `skill/SKILL.md` | Skill instructions injected into Codex's context |
| `skill/agents/openai.yaml` | Skill metadata and implicit invocation policy |

## Uninstall

```bash
rm -rf ~/.codex/skills/memmachine
```

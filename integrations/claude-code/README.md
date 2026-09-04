# MemMachine for Claude Code

Gives Claude Code a durable memory of its own sessions: every turn is captured
into MemMachine's timeline, and three MCP tools read it back.

There is no local daemon and no per-session helper process. The MemMachine
server already holds the loaded embedder and the open stores, which is what a
daemon would have existed to do, and it serves the MCP tools over HTTP — so
Claude Code connects to a URL rather than spawning anything.

## Setup

Run a MemMachine server with the event backend and its MCP HTTP transport:

```bash
memmachine-server                  # the REST API, default :8080
memmachine-mcp-http --port 8081    # the MCP tools
```

Point Claude Code at the tools, in `.mcp.json`:

```json
{
  "mcpServers": {
    "memmachine": { "type": "http", "url": "http://127.0.0.1:8081/mcp" }
  }
}
```

Register the capture hook, in `settings.json`:

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 /path/to/integrations/claude-code/capture_transcript.py"
          }
        ]
      }
    ]
  }
}
```

Then install the skill so the tools come with their usage discipline:

```bash
npx skills add https://github.com/MemMachine/MemMachine \
  --skill integrations/claude-code
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MEMMACHINE_URL` | `http://127.0.0.1:8080` | REST API the capture hook posts to |
| `MM_ORG_ID` | `claude-code` | Organization the timelines live under |
| `MM_PROJ_ID` | derived from `cwd` | Project, so each repository gets its own timeline |
| `MEMMACHINE_STATE_DIR` | `~/.memmachine/claude-code` | Where capture marks are kept |

The hook takes the same values as flags, which is the better choice when the
settings file is shared: `--server`, `--org-id`, `--project-id`, `--state-dir`.

## What the hook stores

The whole transcript, in order — messages, tool calls, tool results, and
reasoning where the transcript records it — because replaying what happened
means reading a contiguous stretch of it.

Each entry carries a `source` in its metadata, and only `user_message` and
`assistant_message` are worth searching. The recall side filters on that, so
tool calls and pasted files are reached by expanding around a message rather
than competing with it for rank.

Text that arrived with a user turn without being typed by the user — hook
output, skill bodies, system reminders, and the session's own compaction
summary — is stored as `injected` and kept out of the index. The compaction
summary is why this matters: compaction continues the same session, so the
summary sits among the very turns it paraphrases, and indexing it would let a
description outrank its own source.

## State

One small file per session, holding how many transcript lines have been
captured. Nothing else is kept between turns.

That is deliberate. The tools here are deliberate recall: you ask for
something, so a result that does not apply is visible as a mismatch against
the purpose you already had. Nothing needs to remember what was already shown.
Suppressing repeats would need that state, and it would also need to be
truncated whenever the session compacts — after a compaction the turns "already
shown" are precisely the ones no longer in context, so the record would mean
the opposite of what it claims. A `PreCompact` hook is where that truncation
would go, if this ever grows a channel that needs it.

The capture mark itself is unaffected by compaction: the transcript is
append-only and compaction continues the same file, so a line count stays
valid for the life of the session.

# Claude Code Hook Events — Verified Reference

Sources: Claude Code hooks documentation (https://docs.anthropic.com/en/docs/claude-code/hooks),
confirmed via `.claude/settings.json` in this repo and the reef reference implementation.

---

## Confirmed Hook Events

| Event | Status | Description |
|---|---|---|
| `SessionStart` | confirmed | Fires when a new Claude Code session begins |
| `UserPromptSubmit` | confirmed | Fires when the user submits a prompt, before Claude responds |
| `PreToolUse` | confirmed | Fires before a tool call is executed |
| `PostToolUse` | confirmed | Fires after a tool call completes |
| `Stop` | confirmed | Fires when Claude stops responding (after final assistant turn) |
| `SessionEnd` | confirmed | Fires when the session ends (window closed / `/exit`) |
| `AssistantResponse` | not available | No dedicated event for assistant message text outside of Stop/SessionEnd |

Notes:
- `Stop` fires at the end of each assistant turn (may fire multiple times per session).
- `SessionEnd` fires once when the session terminates.
- Tool-use events (`PreToolUse`, `PostToolUse`) carry the tool name and input/output but
  are not used by the MemMachine hooks (too granular, not needed for memory ingest).

---

## Stdin Payload Schemas

### UserPromptSubmit

The hook receives a JSON object on stdin with at minimum:

```json
{
  "session_id": "abc123",
  "hook_event_name": "UserPromptSubmit",
  "prompt": "What did I work on last week?"
}
```

Key fields:
- `session_id` — string, unique per Claude Code session window. Use this as the
  MemMachine `--session-id`. It is stable across all hooks in the same session.
- `prompt` — string, the full text of the user's submitted prompt.
- `hook_event_name` — string, always `"UserPromptSubmit"` for this event.

### SessionEnd

```json
{
  "session_id": "abc123",
  "hook_event_name": "SessionEnd",
  "transcript": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Key fields:
- `session_id` — same session_id as UserPromptSubmit hooks in this session.
- `transcript` — array of message objects with `role` and `content` fields.
- The last `assistant` entry in the transcript is the final response.

### Stop (not used, documented for reference)

```json
{
  "session_id": "abc123",
  "hook_event_name": "Stop",
  "stop_reason": "end_turn"
}
```

Does not include transcript content — not suitable for assistant response ingest.

---

## How session_id Is Obtained

`session_id` is present in the **stdin JSON payload** for all hook events. Extract with:

```bash
SESSION_ID=$(jq -r '.session_id // empty' <<< "$INPUT")
```

There is no dedicated environment variable for session_id — it must be parsed from stdin.

---

## Implementation Decisions for CC-02 and CC-03

**CC-02 (assistant response ingest):**
Use `SessionEnd` + `transcript` array. Extract the last assistant-role entry as the
representative assistant response for the session. This captures the final answer without
requiring a `Stop` hook per turn.

**CC-03 (tool output ingest):**
`PostToolUse` events carry individual tool outputs but are per-tool-call and would
generate excessive ingest volume. Decision: do NOT add per-tool-use hooks. Tool outputs
are implicitly captured via the session transcript at `SessionEnd`. If finer granularity
is needed in a future phase, a `PostToolUse` hook can be added.

**Consequence:** `on_session_end.sh` ingests the full final assistant turn from the
transcript. Tool call outputs are not separately ingested — they are included in the
session context but not individually stored as memory episodes.

---

## Installer Note

Hook scripts must be executable. After copying:

```bash
chmod +x on_user_prompt.sh on_session_end.sh
```

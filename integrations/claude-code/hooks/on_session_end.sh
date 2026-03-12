#!/usr/bin/env bash
# on_session_end.sh — Claude Code SessionEnd hook
# Ingests the assistant's final response from the session transcript into MemMachine.
#
# CC-02: assistant response ingest — captured from transcript at session end.
# CC-03: tool output ingest — tool outputs are not ingested separately; they are
#   implicitly present in the session context. Per HOOK-EVENTS.md, PostToolUse hooks
#   are not used to avoid excessive per-call ingest volume.
#
# Install: copy to a stable path and add to ~/.claude/settings.json under SessionEnd.
# Requires: memmachine on $PATH, jq on $PATH.
# Make executable: chmod +x on_session_end.sh

set -euo pipefail

INPUT=$(cat)

SESSION_ID=$(jq -r '.session_id // empty' <<< "$INPUT")

if [ -z "$SESSION_ID" ]; then
  echo "on_session_end.sh: warning: session_id not found in payload, using 'unknown'" >&2
  SESSION_ID="unknown"
fi

# Extract the last assistant turn from the transcript array.
# The transcript field is an array of {role, content} objects.
ASSISTANT_CONTENT=$(jq -r '
  .transcript
  | if type == "array" then
      [ .[] | select(.role == "assistant") ] | last | .content // empty
    else
      empty
    end
' <<< "$INPUT")

if [ -z "$ASSISTANT_CONTENT" ]; then
  # No assistant turns found in transcript — nothing to ingest.
  exit 0
fi

memmachine ingest \
  --role assistant \
  --session-id "$SESSION_ID" \
  --content "$ASSISTANT_CONTENT" \
  || {
    echo "on_session_end.sh: memmachine ingest failed (continuing)" >&2
  }

exit 0

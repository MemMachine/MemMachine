#!/usr/bin/env bash
# on_user_prompt.sh — Claude Code UserPromptSubmit hook
# Ingests the user's prompt into MemMachine memory on every prompt submission.
#
# Install: copy to a stable path and add to ~/.claude/settings.json under UserPromptSubmit.
# Requires: memmachine on $PATH, jq on $PATH.
# Make executable: chmod +x on_user_prompt.sh

set -euo pipefail

INPUT=$(cat)

SESSION_ID=$(jq -r '.session_id // empty' <<< "$INPUT")
PROMPT_TEXT=$(jq -r '.prompt // empty' <<< "$INPUT")

if [ -z "$SESSION_ID" ]; then
  echo "on_user_prompt.sh: warning: session_id not found in payload, using 'unknown'" >&2
  SESSION_ID="unknown"
fi

if [ -z "$PROMPT_TEXT" ]; then
  # Nothing to ingest
  exit 0
fi

memmachine ingest \
  --role user \
  --session-id "$SESSION_ID" \
  --content "$PROMPT_TEXT" \
  || {
    echo "on_user_prompt.sh: memmachine ingest failed (continuing)" >&2
  }

exit 0

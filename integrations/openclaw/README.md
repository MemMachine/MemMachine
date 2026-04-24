# MemMachine OpenClaw Plugin

This plugin integrates OpenClaw with MemMachine to provide persistent,
queryable long-term memory across agent sessions. MemMachine (by MemVerge)
stores interaction history and retrieves high-relevance context at inference
time, enabling response grounding while reducing prompt size and token usage.

The plugin registers the following functions in OpenClaw:

- `memory_search`
- `memory_store`
- `memory_forget`
- `memory_get`

It also registers two CLI functions:

- `search`: Search MemMachine memory.
- `stats`: Retrieve stats from MemMachine.

## Features

### Auto Recall

When auto recall is enabled, the plugin searches episodic and semantic memories
before the agent responds. Matching entries are injected into the context.

### Auto Capture

When auto capture is enabled, the plugin sends each exchange to MemMachine
after the agent responds.

## Setup

### Install from package registry

```bash
openclaw plugins install @memmachine/openclaw-memmachine
```

### Install from local filesystem

```bash
openclaw plugins install ./MemMachine/integrations/openclaw
```

### Install from a packed tarball

```bash
openclaw plugins install ./memmachine-openclaw-memmachine-0.0.0-development.tgz
```

Do not use `openclaw hooks install` for this package. It is an OpenClaw plugin
pack that exports `openclaw.extensions`, not a hook pack that exports
`openclaw.hooks`.

## Platform (MemMachine Cloud)

Get an API key from [MemMachine Cloud](https://console.memmachine.ai).

## Configuration

You can configure the MemMachine plugin in the UI or by editing the
`memmachine` entry in the `openclaw.json` file.

### MemMachine configuration in openclaw.json

Here is a sample `openclaw.json` entry:

```json5
{
  "plugins": {
    "slots": {
      "memory": "openclaw-memmachine"
    },
    "entries": {
      "openclaw-memmachine": {
        "enabled": true,
        "config": {
          "apiKey": "mm-...",
          "baseUrl": "https://api.memmachine.ai",
          "autoCapture": true,
          "autoRecall": true,
          "orgId": "openclaw",
          "projectId": "openclaw",
          "searchThreshold": 0.5,
          "topK": 5,
          "userId": "openclaw"
        }
      }
    }
  }
}
```

### Configuration entries

Here are the required configuration entries:

- `apiKey`: MemMachine API key.
- `baseUrl`: MemMachine API base URL.
- `autoCapture`: Enable automatic memory capture.
- `autoRecall`: Enable automatic memory recall.
- `orgId`: Organization identifier.
- `projectId`: Project identifier.
- `searchThreshold`: Minimum similarity score for recall.
- `topK`: Maximum number of memories to return.
- `userId`: User identifier for memory scoping.

## Memory scoping and `userId`

The `userId` field controls how `autoRecall` scopes its search and how
`autoCapture` tags stored memories. The behaviour depends on whether `userId`
has been set to a real per-human-user value:

### Default `userId` (`"openclaw"`) — session-only recall

When `userId` is omitted or left as the default value `"openclaw"`, the plugin
uses the session's ephemeral `sessionId` as the sole discriminator. Recall is
restricted to memories captured in the **current conversation session** (the
session started by `/new` or `/reset`). Memories from earlier conversations are
not retrieved.

This is the safe default. It prevents memories from one user's questions from
leaking into another user's (or even the same user's later) questions.

### Configured `userId` — cross-session long-term recall

When `userId` is set to a stable, per-human-user identifier — for example the
Slack user ID of the person talking to the bot — `autoRecall` retrieves all
memories previously stored under that `user_id`. This enables genuine long-term
memory: preferences, decisions, and facts persist across separate sessions.

```json5
{
  "config": {
    "userId": "U012AB3CD"  // Slack user ID, Telegram user ID, etc.
  }
}
```

To use this correctly, set `userId` dynamically per user in your integration
layer (e.g. per-channel account config, middleware, or environment variable).
All users sharing the same `userId` value share their memory pool.

### What happens when `sessionId` is unavailable

If OpenClaw does not supply a `sessionId` for a particular hook invocation
(for example in some one-shot or legacy agent run modes), `autoRecall` skips
the search entirely rather than issuing an unscoped query that would match all
stored memories. You will see an `info`-level log line from the plugin when
this occurs.

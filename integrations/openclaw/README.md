# MemMachine OpenClaw Plugin

This plugin provides MemMachine-powered long-term memory for OpenClaw.

The plugin registers the following functions in OpenClaw:
- `memory_search`
- `memory_store`
- `memory_forget`
- `memory_get`

The plugin also registers two CLI functions in OpenClaw:
- `search`: Search MemMachine memory.
- `stats`: Retrieve stats from MemMachine.

## Features

### Auto Recall
If auto recall is enabled, before the agent responds, the plugin searches both
episodic and semantic memories for entries that match the current message and
injects them into context.

### Auto Capture
If auto capture is enabled, after the agent responds, the plugin sends the
exchange to MemMachine. MemMachine stores the most recent message.

## Setup

### Install from package registry

```bash
openclaw plugins install @memmachine/openclaw-memmachine
```

### Install from local file system

```bash
openclaw plugins install ./MemMachine/integrations
cd ./MemMachine/integrations/openclaw && pnpm install
```

## Platform (MemMachine Cloud)

Get an API key from [MemMachine Cloud](https://console.memmachine.ai), then
add this to your `openclaw.json`:

```json5
// plugins.entries
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
```

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
openclaw plugins install ./MemMachine/integrations
cd ./MemMachine/integrations/openclaw && pnpm install
```

## Platform (MemMachine Cloud)

Get an API key from [MemMachine Cloud](https://console.memmachine.ai).

## Configuration

You can configure the MemMachine plugin in the UI or by editing the
`memmachine` entry in the `openclaw.json` file.

### MemMachine configuration in openclaw.json

Here is a sample `openclaw.json` entry:

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

## Setup

```bash
openclaw plugins install @memmachine/openclaw-memmachine
```

### Platform (MemMachine Cloud)

Get an API key from [MemMachine Cloud](https://console.memmachine.ai), then add to your `openclaw.json`:

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
      },

```

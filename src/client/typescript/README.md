# MemMachine REST Client

A unified TypeScript/Node.js SDK for MemMachine RESTful APIs, making it easy to manage memories for users, groups, and agents with a consistent interface.

## Features

- Add, search, and delete memories
- Manage sessions for users, groups, and agents
- Health check for the MemMachine server
- TypeScript types for all API payloads and responses
- Custom error handling with `APIError`

## Installation

```bash
npm install memmachine-client
```

## Usage Example

```typescript
import MemoryClient, { APIError } from 'memmachine-client'

const client = new MemoryClient({ apiKey: 'your-api-key', host: 'https://your-host-url' })

async function run() {
  try {
    await client.addMemory('Test memory', 'message', {
      session_id: 'session_123',
      user_id: ['test_user'],
      producer: 'test_user',
      produced_for: 'test_user'
    })
    const result = await client.searchMemory('Test', { session_id: 'session_123' })
    console.log(result)
  } catch (err) {
    if (err instanceof APIError) {
      // handle error
      console.error(err.message)
    }
  }
}

run()
```

## API Reference

### Main Class

`MemoryClient` — The core client for interacting with MemMachine RESTful APIs

#### Methods

- `addMemory()` — Add a memory to a session
- `addEpisodicMemory()` — Add an episodic memory
- `addProfileMemory()` — Add a profile memory
- `searchMemory()` — Search for memories
- `searchEpisodicMemory()` — Search for episodic memories
- `searchProfileMemory()` — Search for profile memories
- `getSessions()` — List all sessions
- `getUserSessions()` — List sessions for a user
- `getGroupSessions()` — List sessions for a group
- `getAgentSessions()` — List sessions for an agent
- `deleteMemory()` — Delete a memory for a specified session
- `healthCheck()` — Check server health

### Error Handling

`APIError` — Custom error class for API errors

### Types

**Configuration:**

- `ClientOptions`

**Payload and response interfaces:**

- `MemorySession`
- `SessionOptions`
- `MemoryOptions`
- `SearchOptions`
- `SearchResult`
- `HealthStatus`

## Examples

See [basic usage examples](./examples/basic.ts) for practical code demonstrating memory management, session operations, and error handling with the MemMachine REST Client.

To run the example:

```bash
npm run example
```

## Development

1. Install dependencies:

```bash
npm install
```

2. Build the project:

```bash
npm run build
```

3. Clean build files:

```bash
npm run clean
```

4. Run unit tests:

```bash
npm run test
```

## License

MemMachine REST Client is licensed under the [MIT License](./LICENSE).

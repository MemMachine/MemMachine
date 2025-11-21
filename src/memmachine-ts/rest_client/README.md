# MemMachine REST Client

A unified TypeScript/Node.js SDK for MemMachine RESTful APIs, making it easy to manage memories for users, groups, and agents with a consistent interface.

## Features

- Add and search memories
- Manage memory sessions
- Health check for the MemMachine server
- TypeScript types for all API data structures
- Custom error handling with `MemMachineAPIError`

## Installation

```bash
npm install @memmachine/client
```

## Usage Example

```typescript
import MemMachineClient, { MemMachineAPIError } from '@memmachine/client'

async function run() {
  const client = new MemMachineClient({ base_url: 'https://your-base-url' })
  const memory = client.memory({ user_id: 'Paul' })

  try {
    // Adding a memory
    await memory.add('I like pizza and pasta')

    // Searching memories
    const result = await memory.search('What do I like to eat?')
    console.dir(result, { depth: null })
  } catch (err) {
    if (err instanceof MemMachineAPIError) {
      // handle error
      console.error(err.message)
    }
  }
}

run()
```

## API Reference

### Main Classes

`MemMachineClient` — The core client for interacting with MemMachine RESTful APIs

#### Methods

- `memory()` — Create a MemMachineMemory instance for managing memories
- `getSessions()` — List all memory sessions
- `healthCheck()` — Check MemMachine server health

`MemMachineMemory` — Provide methods to manage and interact with the memory session in MemMachine

#### Methods

- `add()` — Add a memory to the memory session
- `search()` — Search for memories
- `getMemoryContext()` — Get the current memory session context

### Error Handling

`MemMachineAPIError` — Custom error class for API errors

### Types

**Configuration Types**

- `ClientOptions` — Options for initializing the main client
- `MemoryContextOptions` — Options for configuring a memory session
- `AddMemoryOptions` — Options for adding a memory
- `SearchMemoryOptions` — Options for searching memories

**Data Types**

- `MemoryContext` — Represents the current memory session context
- `MemorySession` — Represents a memory session record
- `SearchMemoryResult` — Result of a memory search operation
- `HealthStatus` — Server health status response

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

MemMachine REST Client is licensed under the [Apache-2.0 License](../../../LICENSE).

# MemMachine REST Client

A unified TypeScript/Node.js SDK for MemMachine RESTful APIs, providing a consistent interface to manage memories.

## Installation

```bash
npm install @memmachine/client
```

## API Key

For the cloud service, sign in to [MemMachine Platform](https://console.memmachine.ai) to retrieve your API Key.

## Features

- Full memory lifecycle: create, list, search, and delete
- Configurable project management
- Server health monitoring
- Strongly typed API data structures (TypeScript)
- Unified error handling via `MemMachineAPIError`

## Usage Example

```typescript
import MemMachineClient, { MemMachineAPIError } from '@memmachine/client'

async function run() {
  const client = new MemMachineClient({ api_key: 'your_api_key' })

  // Create a MemMachineProject instance
  const project = client.project({ org_id: 'demo_org', project_id: 'demo_project' })

  // Create the project with config options on the MemMachine server if it does not exist
  // await project.create({ description: 'Demo Project' })

  // Create a MemMachineMemory instance for the project
  const memory = project.memory()

  try {
    // Add a memory
    await memory.add('I like pizza and pasta')

    // Search memories
    const result = await memory.search('What do I like to eat?')
    console.dir(result, { depth: null })
  } catch (err) {
    if (err instanceof MemMachineAPIError) {
      // Handle error
      console.error(err.message)
    }
  }
}

run()
```

## API Reference

### Main Classes

#### MemMachineClient

The core client for interacting with MemMachine RESTful APIs.

- `project()` — Create a MemMachineProject instance
- `getProjects()` — List all projects from MemMachine server
- `healthCheck()` — Check MemMachine server health

---

#### MemMachineProject

Provides methods to manage a project and access its associated memory.

- `memory()` — Create a MemMachineMemory instance for managing memories
- `create()` — Create the project in MemMachine
- `get()` — Retrieve the project from MemMachine
- `getEpisodicCount()` — Retrieve the count of episodic memories in the project
- `delete()` — Remove the project from MemMachine

---

#### MemMachineMemory

Provides methods to manage and interact with the memory in MemMachine.

- `add()` — Add a new memory to MemMachine
- `search()` — Search for memories in MemMachine
- `list()` — List memories in the current project
- `delete()` — Delete a memory from MemMachine
- `getContext()` — Retrieve the current memory context

### Error Handling

`MemMachineAPIError` — Custom error class for API errors

### Types

**Configuration Types**

- `ClientOptions` — Options for initializing the main client
- `ProjectContext` — Represents the current project context
- `CreateProjectOptions` — Options for creating a project
- `MemoryContext` — Represents the current memory context
- `AddMemoryOptions` — Options for adding a memory
- `SearchMemoriesOptions` — Options for searching memories
- `ListMemoriesOptions` — Options for listing memories

**Data Types**

- `Project` — Represents a project entity in MemMachine
- `EpisodicMemory` — Represents an episode memory in MemMachine
- `SemanticMemory` — Represents a semantic memory in MemMachine
- `AddMemoryResult` — Result of adding a memory
- `SearchMemoriesResult` — Result of searching memories
- `HealthStatus` — Server health status response

**Value Types**

- `MemoryType` — Allowed values: 'episodic' | 'semantic'
- `MemoryProducerRole` — Allowed values: 'user' | 'assistant' | 'system'

## Examples

See [basic usage examples](../../../examples/ts_rest_client_demo/) for practical code demonstrating memory management, project operations, and error handling with the MemMachine REST Client.

## Development

1. Install dependencies:

```bash
npm install
```

2. Build the project:

```bash
npm run build
```

3. Run unit tests:

```bash
npm run test
```

4. Build docs:

```bash
npm run docs
```

## License

MemMachine REST Client is licensed under the [Apache-2.0 License](../../../LICENSE).

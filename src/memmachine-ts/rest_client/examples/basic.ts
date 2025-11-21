import { MemMachineAPIError } from '@/errors/memmachine-api-error'
import { MemMachineClient } from '@/client/memmachine-client'

// General error handling function
function handleError(error: unknown, context?: string) {
  if (error instanceof MemMachineAPIError) {
    console.error(`[MemMachineAPIError]${context ? ' [' + context + ']' : ''}:`, error.message)
  } else {
    console.error(`[UnknownError]${context ? ' [' + context + ']' : ''}:`, error)
  }
}

async function main() {
  // Initialize the MemoryClient
  const client = new MemMachineClient({
    base_url: 'http://127.0.0.1:8080' // Replace with your actual server address
  })

  const memory = client.memory({ user_id: 'test_user' })

  const memoriesToAdd = [
    {
      content: 'I like pizza and pasta',
      metadata: { type: 'preference', category: 'food' }
    },
    {
      content: 'I work as a software engineer',
      metadata: { type: 'fact', category: 'work' }
    }
  ]

  // Add memories
  console.log('Adding memories...')
  for (const { content, metadata } of memoriesToAdd) {
    try {
      await memory.add(content, { metadata })
      console.log('Added memory:', content)
    } catch (error) {
      handleError(error, 'Adding memory')
    }
  }

  // Search memories
  console.log('Searching memories...')
  const searchQueries = ['What do I like to eat?', 'Tell me about my work']

  for (const query of searchQueries) {
    try {
      const result = await memory.search(query)
      console.log(`Search results for "${query}":`)
      console.dir(result, { depth: null })
    } catch (error) {
      handleError(error, 'Searching memory')
    }
  }

  // Session operations
  const sessions = await client.getSessions()
  console.log('Sessions:')
  console.dir(sessions, { depth: null })

  // Health check
  const healthCheck = await client.healthCheck()
  console.log('Health check:')
  console.dir(healthCheck, { depth: null })
}

main().catch(console.error)

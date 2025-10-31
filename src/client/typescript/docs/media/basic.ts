import { APIError } from '@/api-error'
import { MemoryClient } from '@/memmachine'

// General error handling function
function handleError(error: unknown, context?: string) {
  if (error instanceof APIError) {
    console.error(`[APIError]${context ? ' [' + context + ']' : ''}:`, error.message)
  } else {
    console.error(`[UnknownError]${context ? ' [' + context + ']' : ''}:`, error)
  }
}

async function main() {
  // Initialize the MemoryClient
  const client = new MemoryClient({
    apiKey: 'your-api-key', // Replace with your actual API key
    host: 'http://127.0.0.1:8080' // Replace with your actual server address
  })

  try {
    // Add memories
    await client.addMemory('This is a simple test memory', 'message', {
      session_id: 'session_123',
      user_id: ['test_user'],
      agent_id: ['test_agent'],
      group_id: 'test_group',
      producer: 'test_user',
      produced_for: 'test_agent'
    })
    console.log('Added simple memory')

    await client.addEpisodicMemory('This is an episodic test memory', 'event', {
      session_id: 'session_123',
      user_id: ['test_user'],
      producer: 'test_user',
      produced_for: 'test_agent'
    })
    console.log('Added episodic memory')

    await client.addProfileMemory('This is a profile test memory', 'attribute', {
      session_id: 'session_123',
      user_id: ['test_user'],
      producer: 'test_user',
      produced_for: 'test_agent'
    })
    console.log('Added profile memory')

    // Search operations
    const searchResult = await client.searchMemory('simple test memory', { session_id: 'session_123' })
    console.log('Search result:')
    console.dir(searchResult, { depth: null })

    const episodicSearchResult = await client.searchEpisodicMemory('episodic test memory', {
      session_id: 'session_123'
    })
    console.log('Episodic search result:')
    console.dir(episodicSearchResult, { depth: null })

    const profileSearchResult = await client.searchProfileMemory('profile test memory', {
      session_id: 'session_123'
    })
    console.log('Profile search result:')
    console.dir(profileSearchResult, { depth: null })

    // Session operations
    const sessions = await client.getSessions()
    console.log('Sessions:')
    console.dir(sessions, { depth: null })

    const userSessions = await client.getUserSessions('test_user')
    console.log('User sessions:')
    console.dir(userSessions, { depth: null })

    const groupSessions = await client.getGroupSessions('test_group')
    console.log('Group sessions:')
    console.dir(groupSessions, { depth: null })

    const agentSessions = await client.getAgentSessions('test_agent')
    console.log('Agent sessions:')
    console.dir(agentSessions, { depth: null })

    // Delete a memory
    await client.deleteMemory({ session_id: 'session_123' })
    console.log('Deleted memories for session_id: session_123')

    // Health check
    const healthCheck = await client.healthCheck()
    console.log('Health check:')
    console.dir(healthCheck, { depth: null })
  } catch (error) {
    handleError(error, 'main')
  }

  // Error handling example
  try {
    await client.addMemory('', 'message', {
      session_id: 'session_123',
      producer: 'test_user',
      produced_for: 'test_agent'
    })
  } catch (error) {
    handleError(error, 'addMemory (empty)')
  }
}

main().catch(console.error)

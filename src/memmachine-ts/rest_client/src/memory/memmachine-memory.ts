import type { AxiosInstance } from 'axios'

import { MemMachineAPIError } from '@/errors/memmachine-api-error'
import { handleApiError } from '@/errors/api-error-handler'
import type {
  MemoryContext,
  AddMemoryOptions,
  MemoryType,
  SearchMemoryOptions,
  SearchMemoryResult,
  MemoryContextOptions
} from './memmachine-memory.types'

/**
 * Provides methods to manage and interact with the memory session in MemMachine.
 *
 * @remarks
 * Requires an AxiosInstance for making API requests.
 * Supports adding and searching memories within a specified memory context.
 *
 * Features:
 * - Add a new memory to a session
 * - Search for memories within a session
 * - Retrieve the current memory context
 *
 * @example
 * ```typescript
 * import MemMachineClient from '@memmachine/client'
 *
 * async function run() {
 *   const client = new MemMachineClient({ base_url: 'https://your-base-url' })
 *   const memory = client.memory({ user_id: 'test_user' })
 *
 *   // Adding a memory
 *   await memory.add('This is a simple memory', { episode_type: 'note' })
 *
 *   // Searching memories
 *   const result = await memory.search('Show a simple memory', { limit: 5 })
 *   console.dir(result, { depth: null })
 * }
 *
 * run()
 * ```
 *
 * @param client - AxiosInstance for API communication.
 * @param contextOptions - Options to configure the memory context, see {@link MemoryContextOptions}.
 */
export class MemMachineMemory {
  client: AxiosInstance
  memoryContext: MemoryContext

  constructor(client: AxiosInstance, contextOptions: MemoryContextOptions) {
    this.client = client

    const { session_id = crypto.randomUUID(), user_id, group_id, agent_id } = contextOptions

    // user_ids
    let user_ids: string[] = []
    if (user_id === undefined || user_id === null) {
      user_ids = []
    } else if (Array.isArray(user_id)) {
      user_ids = user_id.filter(Boolean)
    } else {
      user_ids = [user_id]
    }
    if (user_ids.length === 0) {
      throw new MemMachineAPIError('At least one user id must be provided in MemoryContextOptions')
    }

    // agent_ids
    let agent_ids: string[] = []
    if (agent_id === undefined || agent_id === null) {
      agent_ids = []
    } else if (Array.isArray(agent_id)) {
      agent_ids = agent_id.filter(Boolean)
    } else {
      agent_ids = [agent_id]
    }

    // group_id
    const resolvedGroupId = group_id && group_id.trim() ? group_id : user_ids[0]

    this.memoryContext = {
      session_id,
      user_ids,
      group_id: resolvedGroupId || '',
      agent_ids
    }

    this._setMemoryContextHeaders()
  }

  /**
   * Adds a new memory to the MemMachine memory session.
   *
   * @param content - The content of the memory to be added.
   * @param options - Additional options for adding the memory.
   * @returns A promise that resolves when the memory is successfully added.
   * @throws {@link MemMachineAPIError} if the API request fails.
   */
  add(content: string, options?: AddMemoryOptions): Promise<null> {
    return this._addMemory(content, options || {})
  }

  /**
   * Searches memories within the MemMachine memory session.
   *
   * @param query - The search query string.
   * @param options - Additional options for searching memories.
   * @returns A promise that resolves to the search results.
   * @throws {@link MemMachineAPIError} if the API request fails.
   */
  search(query: string, options?: SearchMemoryOptions): Promise<SearchMemoryResult> {
    return this._searchMemory(query, options || {})
  }

  /**
   * Retrieves the current memory context.
   *
   * @returns The memory context object.
   */
  getMemoryContext(): MemoryContext {
    return this.memoryContext
  }

  private _setMemoryContextHeaders() {
    const { session_id, user_ids, group_id, agent_ids } = this.memoryContext
    if (session_id && session_id.trim()) {
      this.client.defaults.headers.common['session-id'] = session_id
    }
    if (user_ids && user_ids.length > 0) {
      this.client.defaults.headers.common['user-id'] = user_ids.join(',')
    }
    if (group_id && group_id.trim()) {
      this.client.defaults.headers.common['group-id'] = group_id
    }
    if (agent_ids && agent_ids.length > 0) {
      this.client.defaults.headers.common['agent-id'] = agent_ids.join(',')
    }
  }

  private async _addMemory(
    content: string,
    options: AddMemoryOptions,
    memoryType?: MemoryType
  ): Promise<null> {
    const { producer, produced_for, episode_type = 'message', metadata } = options
    const { user_ids } = this.memoryContext

    const payload = {
      producer: producer || user_ids[0],
      produced_for: produced_for || user_ids[0],
      episode_content: content,
      episode_type,
      metadata: metadata ?? undefined
    }

    let url: string
    switch (memoryType) {
      case 'episodic':
        url = '/v1/memories/episodic'
        break
      case 'profile':
        url = '/v1/memories/profile'
        break
      default:
        url = '/v1/memories'
    }

    try {
      const response = await this.client.post(url, payload)
      return response.data
    } catch (error: unknown) {
      handleApiError(
        error,
        `Failed to add ${memoryType ? memoryType + ' ' : ''}memory with payload: ${JSON.stringify(payload)}`
      )
    }
  }

  private async _searchMemory(
    query: string,
    options: SearchMemoryOptions,
    memoryType?: MemoryType
  ): Promise<SearchMemoryResult> {
    const { filter, limit } = options

    if (!query || !query.trim()) {
      throw new MemMachineAPIError('Search query must be a non-empty string')
    }

    const payload = {
      query,
      filter: filter ?? undefined,
      limit: limit ?? undefined
    }

    let url: string
    switch (memoryType) {
      case 'episodic':
        url = '/v1/memories/episodic/search'
        break
      case 'profile':
        url = '/v1/memories/profile/search'
        break
      default:
        url = '/v1/memories/search'
    }

    try {
      const response = await this.client.post(url, payload)
      return response.data
    } catch (error: unknown) {
      handleApiError(
        error,
        `Failed to search ${memoryType ? memoryType + ' ' : ''}memory with payload: ${JSON.stringify(payload)}`
      )
    }
  }
}

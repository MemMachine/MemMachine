import type { AxiosInstance } from 'axios'

import { handleAPIError, MemMachineAPIError } from '@/errors'
import type { ProjectContext } from '@/project'
import type {
  MemoryContext,
  AddMemoryOptions,
  MemoryType,
  SearchMemoryOptions,
  SearchMemoryResult,
  AddMemoryResult
} from './memmachine-memory.types'

/**
 * Provides methods to manage and interact with the memory in MemMachine.
 *
 * @remarks
 * - Requires an AxiosInstance for making API requests.
 * - Requires a ProjectContext to specify the project scope.
 * - Supports adding and searching memories within a specified project and memory context.
 *
 * Features:
 * - Add a new memory to MemMachine
 * - Search for memories within MemMachine
 * - Retrieve the current memory context
 *
 * @example
 * ```typescript
 * import MemMachineClient from '@memmachine/client'
 *
 * async function run() {
 *   const client = new MemMachineClient({ base_url: 'https://your-base-url' })
 *   const project = client.project({ org_id: 'your_org_id', project_id: 'your_project_id' })
 *   const memory = project.memory()
 *
 *   // Add a memory
 *   await memory.add('This is a simple memory', { episode_type: 'note' })
 *
 *   // Search memories
 *   const result = await memory.search('Show a simple memory', { limit: 5 })
 *   console.dir(result, { depth: null })
 *
 *   // Delete a memory
 *   await memory.delete('memory_id', 'episodic')
 *
 *  // Get current memory context
 *  const context = memory.getContext()
 *  console.log(context)
 * }
 *
 * run()
 * ```
 *
 * @param client - AxiosInstance for API communication.
 * @param projectContext - Options to configure the project context, see {@link ProjectContext}.
 * @param memoryContext - Options to configure the memory context, see {@link MemoryContext}.
 */
export class MemMachineMemory {
  client: AxiosInstance
  projectContext: ProjectContext
  memoryContext: MemoryContext

  constructor(client: AxiosInstance, projectContext: ProjectContext, memoryContext?: MemoryContext) {
    this.client = client
    this.projectContext = projectContext
    this.memoryContext = memoryContext ?? {}
  }

  /**
   * Adds a new memory to MemMachine.
   *
   * @param content - The content of the memory to be added.
   * @param options - Additional options for adding the memory.
   * @param type - The type of memory to add.
   * @returns A promise that resolves when the memory is successfully added.
   * @throws {@link MemMachineAPIError} if the API request fails.
   */
  add(content: string, options?: AddMemoryOptions, type?: MemoryType): Promise<AddMemoryResult> {
    return this._addMemory(content, options, type)
  }

  /**
   * Searches memories within MemMachine.
   *
   * @param query - The search query string.
   * @param options - Additional options for searching memories.
   * @returns A promise that resolves to the search results.
   * @throws {@link MemMachineAPIError} if the API request fails.
   */
  search(query: string, options?: SearchMemoryOptions): Promise<SearchMemoryResult> {
    return this._searchMemory(query, options)
  }

  /**
   * Deletes a memory from MemMachine.
   *
   * @param id - The unique identifier of the memory to be deleted.
   * @param type - The type of memory to delete.
   * @returns A promise that resolves when the memory is successfully deleted.
   * @throws {@link MemMachineAPIError} if the API request fails.
   */
  delete(id: string, type: MemoryType): Promise<void> {
    return this._deleteMemory(id, type)
  }

  /**
   * Retrieves the current memory context.
   *
   * @returns The combined project and memory context.
   */
  getContext(): ProjectContext & MemoryContext {
    return {
      ...this.projectContext,
      ...this.memoryContext
    }
  }

  private async _addMemory(
    content: string,
    options?: AddMemoryOptions,
    memoryType?: MemoryType
  ): Promise<AddMemoryResult> {
    if (memoryType) {
      this._validateMemoryType(memoryType)
    }

    let { producer, role = 'user', produced_for, episode_type, metadata = {} } = options ?? {}
    const { user_id, agent_id } = this.memoryContext

    if (!producer && user_id?.length) {
      producer = Array.isArray(user_id) ? user_id[0] : user_id
    }

    if (!produced_for && agent_id?.length) {
      produced_for = Array.isArray(agent_id) ? agent_id[0] : agent_id
    }

    const payload = {
      ...this.projectContext,
      messages: [
        {
          content,
          producer: producer || 'user',
          produced_for: produced_for || 'agent',
          timestamp: new Date().toISOString(),
          role,
          metadata: {
            ...metadata,
            ...this.memoryContext,
            episode_type
          }
        }
      ]
    }

    let url: string
    switch (memoryType) {
      case 'episodic':
        url = '/api/v2/memories/episodic/add'
        break
      case 'semantic':
        url = '/api/v2/memories/semantic/add'
        break
      default:
        url = '/api/v2/memories'
    }

    try {
      const response = await this.client.post(url, payload)
      return response.data
    } catch (error: unknown) {
      handleAPIError(
        error,
        `Failed to add ${memoryType ? memoryType + ' ' : ''}memory with payload: ${JSON.stringify(payload)}`
      )
    }
  }

  private async _searchMemory(query: string, options?: SearchMemoryOptions): Promise<SearchMemoryResult> {
    if (!query || !query.trim()) {
      throw new MemMachineAPIError('Search query must be a non-empty string')
    }

    const { filter = {}, limit = 10, types = ['episodic', 'semantic'] } = options ?? {}

    const payload = {
      ...this.projectContext,
      query,
      filter: JSON.stringify(filter),
      top_k: limit,
      types
    }

    try {
      const response = await this.client.post('/api/v2/memories/search', payload)
      return response.data
    } catch (error: unknown) {
      handleAPIError(error, `Failed to search memory with payload: ${JSON.stringify(payload)}`)
    }
  }

  private async _deleteMemory(id: string, memoryType: MemoryType): Promise<void> {
    if (!id || !id.trim()) {
      throw new MemMachineAPIError('Memory ID must be a non-empty string')
    }

    this._validateMemoryType(memoryType)

    const urlMap: Record<MemoryType, string> = {
      episodic: '/api/v2/memories/episodic/delete',
      semantic: '/api/v2/memories/semantic/delete'
    }

    const payload = {
      ...this.projectContext,
      ...(memoryType === 'episodic' ? { episodic_id: id } : {}),
      ...(memoryType === 'semantic' ? { semantic_id: id } : {})
    }

    try {
      await this.client.post(urlMap[memoryType], payload)
    } catch (error: unknown) {
      handleAPIError(error, `Failed to delete ${memoryType} memory with payload: ${JSON.stringify(payload)}`)
    }
  }

  private _validateMemoryType(type: MemoryType): void {
    const validTypes: MemoryType[] = ['episodic', 'semantic']
    if (!validTypes.includes(type)) {
      throw new MemMachineAPIError(`Invalid memory type: ${type}. Valid types are: ${validTypes.join(', ')}`)
    }
  }
}

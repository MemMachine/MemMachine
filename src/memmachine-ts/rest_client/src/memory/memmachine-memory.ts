import type { AxiosInstance } from 'axios'

import { handleAPIError, MemMachineAPIError } from '@/errors'
import type { ProjectContext } from '@/project'
import type {
  MemoryContext,
  AddMemoryOptions,
  MemoryType,
  SearchMemoriesOptions,
  SearchMemoriesResult,
  AddMemoryResult,
  ListMemoriesOptions
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
 *   const client = new MemMachineClient({ api_key: 'your_api_key' })
 *   const project = client.project({ org_id: 'your_org_id', project_id: 'your_project_id' })
 *   const memory = project.memory()
 *
 *   // Add a memory
 *   await memory.add('This is a simple memory', { episode_type: 'note' })
 *
 *   // Search memories
 *   const result = await memory.search('Show a simple memory', { top_k: 5 })
 *   console.dir(result, { depth: null })
 *
 *  // List memories
 *  const listResult = await memory.list({ page_size: 5, page_num: 0 })
 *  console.dir(listResult, { depth: null })
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
   * @returns A promise that resolves when the memory is successfully added.
   * @throws {@link MemMachineAPIError} if the API request fails.
   */
  add(content: string, options?: AddMemoryOptions): Promise<AddMemoryResult> {
    return this._addMemory(content, options)
  }

  /**
   * Searches memories within MemMachine.
   *
   * @param query - The search query string.
   * @param options - Additional options for searching memories.
   * @returns A promise that resolves to the search results.
   * @throws {@link MemMachineAPIError} if the API request fails.
   */
  search(query: string, options?: SearchMemoriesOptions): Promise<SearchMemoriesResult> {
    return this._searchMemories(query, options)
  }

  /**
   * Lists memories within MemMachine for the current project.
   *
   * @param options - Additional options for listing memories.
   * @returns A promise that resolves to the search results.
   * @throws {@link MemMachineAPIError} if the API request fails.
   */
  list(options?: ListMemoriesOptions): Promise<SearchMemoriesResult> {
    return this._listMemories(options)
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

  /**
   * Implements the logic to add a new memory to MemMachine.
   *
   * @param content - The content of the memory to be added.
   * @param options - Additional options for adding the memory.
   * @returns A promise that resolves when the memory is successfully added.
   * @throws {@link MemMachineAPIError} if the API request fails.
   */
  private async _addMemory(content: string, options?: AddMemoryOptions): Promise<AddMemoryResult> {
    const {
      producer,
      role = 'user',
      produced_for,
      episode_type,
      timestamp,
      metadata = {},
      types = ['episodic', 'semantic']
    } = options ?? {}

    this._validateMemoryRole(role)

    const isoTimestamp = this._parseToIsoTimestamp(timestamp)

    const payload = {
      ...this.projectContext,
      types,
      messages: [
        {
          content,
          producer,
          role,
          produced_for,
          episode_type,
          timestamp: isoTimestamp,
          metadata: {
            ...this.memoryContext,
            ...metadata
          }
        }
      ]
    }

    try {
      const response = await this.client.post('/memories', payload)
      return response.data
    } catch (error: unknown) {
      handleAPIError(error, `Failed to add memory with payload: ${JSON.stringify(payload)}`)
    }
  }

  /**
   * Implements the logic to search memories within MemMachine.
   *
   * @param query - The search query string.
   * @param options - Additional options for searching memories.
   * @returns A promise that resolves to the search results.
   * @throws {@link MemMachineAPIError} if the API request fails.
   */
  private async _searchMemories(
    query: string,
    options?: SearchMemoriesOptions
  ): Promise<SearchMemoriesResult> {
    if (!query || !query.trim()) {
      throw new MemMachineAPIError('Search query must be a non-empty string')
    }

    const { top_k = 10, filter = '', types = ['episodic', 'semantic'] } = options ?? {}

    const payload = {
      ...this.projectContext,
      query,
      top_k,
      filter,
      types
    }

    try {
      const response = await this.client.post('/memories/search', payload)
      return response.data
    } catch (error: unknown) {
      handleAPIError(error, `Failed to search memories with payload: ${JSON.stringify(payload)}`)
    }
  }

  /**
   * Implements the logic to list memories within MemMachine.
   *
   * @param options - Additional options for listing memories.
   * @returns A promise that resolves to the search results.
   * @throws {@link MemMachineAPIError} if the API request fails.
   */
  private async _listMemories(options?: ListMemoriesOptions): Promise<SearchMemoriesResult> {
    const { page_size = 10, page_num = 0, filter = '', type = 'episodic' } = options ?? {}

    const payload = {
      ...this.projectContext,
      page_size,
      page_num,
      filter,
      type
    }

    try {
      const response = await this.client.post('/memories/list', payload)
      return response.data
    } catch (error: unknown) {
      handleAPIError(error, `Failed to list memories with payload: ${JSON.stringify(payload)}`)
    }
  }

  /**
   * Implements the logic to delete a memory from MemMachine.
   *
   * @param id - The unique identifier of the memory to be deleted.
   * @param memoryType - The type of memory to delete.
   * @returns A promise that resolves when the memory is successfully deleted.
   * @throws {@link MemMachineAPIError} if the API request fails.
   */
  private async _deleteMemory(id: string, memoryType: MemoryType): Promise<void> {
    if (!id || !id.trim()) {
      throw new MemMachineAPIError('Memory ID must be a non-empty string')
    }

    this._validateMemoryType(memoryType)

    const urlMap: Record<MemoryType, string> = {
      episodic: '/memories/episodic/delete',
      semantic: '/memories/semantic/delete'
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

  /**
   * Validates the memory type.
   *
   * @param type - The memory type to validate.
   * @throws {MemMachineAPIError} If the memory type is invalid.
   */
  private _validateMemoryType(type: MemoryType): void {
    const validTypes: MemoryType[] = ['episodic', 'semantic']
    if (!validTypes.includes(type)) {
      throw new MemMachineAPIError(`Invalid memory type: ${type}. Valid types are: ${validTypes.join(', ')}`)
    }
  }

  /**
   * Validates the memory producer role.
   *
   * @param role - The role to validate.
   * @throws {MemMachineAPIError} If the role is invalid.
   */
  private _validateMemoryRole(role: string): void {
    const validRoles = ['user', 'system', 'assistant']
    if (!validRoles.includes(role)) {
      throw new MemMachineAPIError(`Invalid memory role: ${role}. Valid roles are: ${validRoles.join(', ')}`)
    }
  }

  /**
   * Converts a timestamp string to ISO format if valid, otherwise returns current time in ISO format.
   *
   * @param timestamp - The timestamp string to convert.
   * @returns ISO 8601 formatted string.
   */
  private _parseToIsoTimestamp(timestamp?: string): string {
    if (timestamp) {
      const parsed = Date.parse(timestamp)
      return !isNaN(parsed) ? new Date(parsed).toISOString() : new Date().toISOString()
    }
    return new Date().toISOString()
  }
}

import { type AxiosInstance } from 'axios'
import axios from 'axios'

import { APIError } from './api-error'
import type {
  ClientOptions,
  HealthStatus,
  MemoryOptions,
  MemorySession,
  SearchOptions,
  SearchResult,
  SessionOptions
} from './memmachine.types'

/**
 * Main API client for interacting with the MemMachine RESTful service.
 *
 * Provides unified methods for memory management, session operations, and health checks.
 *
 * @remarks
 * All API calls require a valid {@link ClientOptions.apiKey} and {@link ClientOptions.host}.
 *
 * Features:
 * - Add, search, and delete memories
 * - Manage sessions for users, groups, and agents
 * - Health check for the MemMachine server
 *
 * @example
 * ```typescript
 * import MemoryClient, { APIError } from 'memmachine-client';
 *
 * const client = new MemoryClient({ apiKey: 'your-api-key', host: 'https://your-host-url' });
 *
 * try {
 *   await client.addMemory('Test memory', 'message', {
 *     session_id: 'session_123',
 *     user_id: ['test_user'],
 *     producer: 'test_user',
 *     produced_for: 'test_agent'
 *   });
 *   const result = await client.searchMemory('Test', { session_id: 'session_123' });
 *   console.log(result);
 * } catch (err) {
 *   if (err instanceof APIError) {
 *     // handle error
 *   }
 * }
 * ```
 *
 * @param options - Configuration options for the client, see {@link ClientOptions}.
 */
export class MemoryClient {
  apiKey: string
  host: string
  client: AxiosInstance

  constructor(options: ClientOptions) {
    this.apiKey = options.apiKey
    this.host = options.host

    this.client = axios.create({
      baseURL: this.host,
      headers: { Authorization: `Token ${this.apiKey}` },
      timeout: options.timeout || 60000
    })

    this._validateApiKey()
  }

  /**
   * Adds a new memory to the MemMachine server.
   *
   * @param message - The memory content to store.
   * @param message_type - The type of the memory (e.g., 'dialog', 'summary').
   * @param options - Memory options.
   * @throws {@link APIError} if the request fails or required fields are missing.
   */
  async addMemory(message: string, message_type: string, options: MemoryOptions) {
    return await this._addMemory('/v1/memories/', message, message_type, options)
  }

  /**
   * Adds a new episodic memory to the MemMachine server.
   *
   * @param message - The episodic memory content to store.
   * @param message_type - The type of the episodic memory (e.g., 'event', 'dialog').
   * @param options - Memory options.
   * @throws {@link APIError} if the request fails or required fields are missing.
   */
  async addEpisodicMemory(message: string, message_type: string, options: MemoryOptions) {
    return await this._addMemory('/v1/memories/episodic/', message, message_type, options, 'episodic')
  }

  /**
   * Adds a new profile memory to the MemMachine server.
   *
   * @param message - The profile memory content to store.
   * @param message_type - The type of the profile memory (e.g., 'attribute', 'summary').
   * @param options - Memory options.
   * @throws {@link APIError} if the request fails or required fields are missing.
   */
  async addProfileMemory(message: string, message_type: string, options: MemoryOptions) {
    return await this._addMemory('/v1/memories/profile/', message, message_type, options, 'profile')
  }

  /**
   * Searches for memories in the MemMachine server using a query string and additional options.
   *
   * @param query - The search query string.
   * @param options - Search options.
   * @returns SearchResult object containing status and content.
   * @throws {@link APIError} if the request fails or required fields are missing.
   */
  async searchMemory(query: string, options: SearchOptions): Promise<SearchResult> {
    return await this._searchMemory('/v1/memories/search/', query, options)
  }

  /**
   * Searches for episodic memories in the MemMachine server using a query string and additional options.
   *
   * @param query - The search query string.
   * @param options - Search options.
   * @returns SearchResult object containing status and content.
   * @throws {@link APIError} if the request fails or required fields are missing.
   */
  async searchEpisodicMemory(query: string, options: SearchOptions): Promise<SearchResult> {
    return await this._searchMemory('/v1/memories/episodic/search/', query, options, 'episodic')
  }

  /**
   * Searches for profile memories in the MemMachine server using a query string and additional options.
   *
   * @param query - The search query string.
   * @param options - Search options.
   * @returns SearchResult object containing status and content.
   * @throws {@link APIError} if the request fails or required fields are missing.
   */
  async searchProfileMemory(query: string, options: SearchOptions): Promise<SearchResult> {
    return await this._searchMemory('/v1/memories/profile/search/', query, options, 'profile')
  }

  /**
   * Deletes memories from the MemMachine server for a specified session.
   *
   * @param options - Session options.
   * @throws {@link APIError} if the request fails or required fields are missing.
   */
  async deleteMemory(options: SessionOptions) {
    const { session_id, user_id, group_id, agent_id } = options

    this._checkRequiredFields([{ value: session_id, name: 'session_id' }])

    const payload = {
      session: {
        session_id,
        user_id: user_id || null,
        group_id: group_id || null,
        agent_id: agent_id || null
      }
    }

    try {
      const response = await this.client.delete('/v1/memories/', {
        data: payload
      })
      return response.data
    } catch (error: unknown) {
      this._handleApiError(error, `Failed to delete memory with payload: ${JSON.stringify(payload)}`)
    }
  }

  /**
   * Retrieves all memory sessions from the MemMachine server.
   *
   * @returns An array of MemorySession objects.
   * @throws {@link APIError} if the request fails.
   */
  async getSessions(): Promise<Array<MemorySession>> {
    return await this._getSessions('/v1/sessions/')
  }

  /**
   * Retrieves all memory sessions for a specific user from the MemMachine server.
   *
   * @param user_id - The user ID to retrieve sessions for.
   * @returns An array of MemorySession objects for the user.
   * @throws {@link APIError} if the request fails or user_id is missing.
   */
  async getUserSessions(user_id: string): Promise<Array<MemorySession>> {
    this._checkRequiredFields([{ value: user_id, name: 'user_id' }])
    return await this._getSessions(`/v1/users/${user_id}/sessions/`, 'user')
  }

  /**
   * Retrieves all memory sessions for a specific group from the MemMachine server.
   *
   * @param group_id - The group ID to retrieve sessions for.
   * @returns An array of MemorySession objects for the group.
   * @throws {@link APIError} if the request fails or group_id is missing.
   */
  async getGroupSessions(group_id: string): Promise<Array<MemorySession>> {
    this._checkRequiredFields([{ value: group_id, name: 'group_id' }])
    return await this._getSessions(`/v1/groups/${group_id}/sessions/`, 'group')
  }

  /**
   * Retrieves all memory sessions for a specific agent from the MemMachine server.
   *
   * @param agent_id - The agent ID to retrieve sessions for.
   * @returns An array of MemorySession objects for the agent.
   * @throws {@link APIError} if the request fails or agent_id is missing.
   */
  async getAgentSessions(agent_id: string): Promise<Array<MemorySession>> {
    this._checkRequiredFields([{ value: agent_id, name: 'agent_id' }])
    return await this._getSessions(`/v1/agents/${agent_id}/sessions/`, 'agent')
  }

  /**
   * Checks the health status of the MemMachine server.
   *
   * @returns The server status information.
   * @throws {@link APIError} if the request fails.
   */
  async healthCheck(): Promise<HealthStatus> {
    try {
      const response = await this.client.get('/health/')
      return response.data
    } catch (error: unknown) {
      this._handleApiError(error, 'Failed to check health status')
    }
  }

  private _validateApiKey() {
    if (typeof this.apiKey !== 'string' || !this.apiKey.trim()) {
      throw new APIError('Memmachine API key must be a non-empty string')
    }
  }

  private _handleApiError(error: unknown, message: string): never {
    if (error instanceof Error) {
      throw new APIError(`${message}: ${error.message}`)
    }
    throw new APIError(`${message}: ${JSON.stringify(error)}`)
  }

  private _checkRequiredFields(fields: Array<{ value: string; name: string }>) {
    for (const { value, name } of fields) {
      if (value == null || value.trim() === '') {
        throw new APIError(`${name} field missing`)
      }
    }
  }

  private async _addMemory(
    url: string,
    message: string,
    message_type: string,
    options: MemoryOptions,
    memoryType?: 'episodic' | 'profile'
  ) {
    const { session_id, user_id, group_id, agent_id, producer, produced_for, metadata } = options

    this._checkRequiredFields([
      { value: message, name: 'message' },
      { value: message_type, name: 'message_type' },
      { value: session_id, name: 'session_id' },
      { value: producer, name: 'producer' },
      { value: produced_for, name: 'produced_for' }
    ])

    const payload = {
      session: {
        session_id,
        user_id: user_id || null,
        group_id: group_id || null,
        agent_id: agent_id || null
      },
      producer,
      produced_for,
      episode_content: message,
      episode_type: message_type,
      metadata: metadata || null
    }

    try {
      const response = await this.client.post(url, payload)
      return response.data
    } catch (error: unknown) {
      this._handleApiError(
        error,
        `Failed to add ${memoryType ? memoryType + ' ' : ''}memory with payload: ${JSON.stringify(payload)}`
      )
    }
  }

  private async _searchMemory(
    url: string,
    query: string,
    options: SearchOptions,
    memoryType?: 'episodic' | 'profile'
  ): Promise<SearchResult> {
    const { session_id, user_id, group_id, agent_id, filter, limit } = options

    this._checkRequiredFields([
      { value: query, name: 'query' },
      { value: session_id, name: 'session_id' }
    ])

    const payload = {
      session: {
        session_id,
        user_id: user_id || null,
        group_id: group_id || null,
        agent_id: agent_id || null
      },
      query,
      filter: filter || null,
      limit: limit || null
    }

    try {
      const response = await this.client.post(url, payload)
      return response.data
    } catch (error: unknown) {
      this._handleApiError(
        error,
        `Failed to search ${memoryType ? memoryType + ' ' : ''}memory with payload: ${JSON.stringify(payload)}`
      )
    }
  }

  private async _getSessions(
    url: string,
    session_type?: 'user' | 'group' | 'agent'
  ): Promise<Array<MemorySession>> {
    try {
      const response = await this.client.get(url)
      return response.data.sessions || []
    } catch (error: unknown) {
      this._handleApiError(
        error,
        `Failed to fetch ${session_type ? session_type + ' ' : ''}sessions from url: ${url}`
      )
    }
  }
}

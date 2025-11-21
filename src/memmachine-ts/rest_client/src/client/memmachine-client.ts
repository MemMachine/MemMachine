import { type AxiosInstance } from 'axios'
import axios from 'axios'
import axiosRetry from 'axios-retry'

import { handleApiError } from '@/errors/api-error-handler'
import { MemMachineAPIError } from '@/errors/memmachine-api-error'
import { MemMachineMemory, type MemoryContextOptions } from '@/memory'
import { VERSION } from '@/version'
import type { ClientOptions, HealthStatus, MemorySession, MemorySessionType } from './memmachine-client.types'

/**
 * Main API client for interacting with the MemMachine RESTful service.
 *
 * Provides unified methods for memory management, session operations, and health checks.
 *
 * @remarks
 * Requires {@link ClientOptions.base_url};
 * {@link ClientOptions.api_key},
 * {@link ClientOptions.timeout} (default: 60000 ms),
 * and {@link ClientOptions.max_retries} (default: 3) are optional.
 *
 * Features:
 * - Manage memories via MemMachineMemory instances
 * - Perform server health checks
 * - List and manage memory sessions
 *
 * @example
 * ```typescript
 * import MemMachineClient from '@memmachine/client'
 *
 * async function run() {
 *   const client = new MemMachineClient({ base_url: 'https://your-base-url' })
 *   const memory = client.memory({ user_id: 'test_user' })
 *   console.log(memory.getMemoryContext())
 *
 *   const healthStatus = await client.healthCheck()
 *   console.dir(healthStatus, { depth: null })
 *
 *   const sessions = await client.getSessions()
 *   console.dir(sessions, { depth: null })
 * }
 *
 * run()
 * ```
 *
 * @param options - Configuration options for the client, see {@link ClientOptions}.
 */
export class MemMachineClient {
  client: AxiosInstance

  constructor(options: ClientOptions) {
    const { base_url, api_key, timeout, max_retries } = options

    if (typeof base_url !== 'string' || !base_url.trim()) {
      throw new MemMachineAPIError('Base URL must be a non-empty string')
    }

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'user-agent': `memmachine-ts-client/${VERSION}`
    }
    if (api_key) {
      headers['Authorization'] = `Token ${api_key}`
    }

    this.client = axios.create({
      baseURL: base_url,
      headers,
      timeout: timeout ?? 60000
    })

    axiosRetry(this.client, {
      retries: max_retries ?? 3,
      retryDelay: (retryCount, error) => axiosRetry.exponentialDelay(retryCount, error, 1000),
      retryCondition: error =>
        axiosRetry.isNetworkOrIdempotentRequestError(error) ||
        (typeof error?.response?.status === 'number' &&
          [429, 500, 502, 503, 504].includes(error.response.status))
    })
  }

  /**
   * Creates a MemMachineMemory instance for managing memories.
   *
   * @param contextOptions - Options to configure the memory context.
   * @returns A MemMachineMemory instance for memory operations.
   */
  memory(contextOptions: MemoryContextOptions): MemMachineMemory {
    return new MemMachineMemory(this.client, contextOptions)
  }

  /**
   * Checks the health status of the MemMachine server.
   *
   * @returns A promise that resolves to the server status information.
   * @throws {@link MemMachineAPIError} if the request fails.
   */
  async healthCheck(): Promise<HealthStatus> {
    try {
      const response = await this.client.get('/health/')
      return response.data
    } catch (error: unknown) {
      handleApiError(error, 'Failed to check health status')
    }
  }

  /**
   * Retrieves all memory sessions from the MemMachine server.
   *
   * @returns A promise that resolves to an array of memory sessions.
   * @throws {@link MemMachineAPIError} if the request fails.
   */
  async getSessions(): Promise<Array<MemorySession>> {
    return await this._getSessions('/v1/sessions/')
  }

  private async _getSessions(url: string, session_type?: MemorySessionType): Promise<Array<MemorySession>> {
    try {
      const response = await this.client.get(url)
      return response.data.sessions || []
    } catch (error: unknown) {
      handleApiError(
        error,
        `Failed to fetch ${session_type ? session_type + ' ' : ''}sessions from url: ${url}`
      )
    }
  }
}

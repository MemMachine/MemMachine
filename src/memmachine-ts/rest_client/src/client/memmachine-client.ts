import axios, { type AxiosInstance } from 'axios'
import axiosRetry from 'axios-retry'

import { handleAPIError, MemMachineAPIError } from '@/errors'
import { MemMachineProject, type Project, type ProjectContext } from '@/project'
import { VERSION } from '@/version'
import type { ClientOptions, HealthStatus } from './memmachine-client.types'

/**
 * Main API client for interacting with the MemMachine RESTful service.
 *
 * Provides a unified API for memory management, project operations, and server health checks.
 *
 * @remarks
 * - Requires {@link ClientOptions.base_url}.
 * - Optional:
 *   {@link ClientOptions.api_key},
 *   {@link ClientOptions.timeout} (default: 60000 ms),
 *   {@link ClientOptions.max_retries} (default: 3).
 *
 * Features:
 * - Manage memories using MemMachineMemory instances
 * - Manage projects using MemMachineProject instances
 * - List projects from the MemMachine server
 * - Perform server health checks
 *
 * @example
 * ```typescript
 * import MemMachineClient from '@memmachine/client'
 *
 * async function run() {
 *   const client = new MemMachineClient({ base_url: 'https://your-base-url' })
 *   const project = client.project({ org_id: 'your_org_id', project_id: 'your_project_id' })
 *   const memory = project.memory()
 *   console.log(memory.getContext())
 *
 *   const projects = await client.getProjects()
 *   console.dir(projects, { depth: null })
 *
 *   const healthStatus = await client.healthCheck()
 *   console.dir(healthStatus, { depth: null })
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
   * Creates a MemMachineProject instance for managing a specific project.
   *
   * @param projectContext - Context options for the project.
   * @returns A MemMachineProject instance.
   */
  project(projectContext: ProjectContext): MemMachineProject {
    return new MemMachineProject(this.client, projectContext)
  }

  /**
   * Retrieves a list of all projects accessible to the client.
   *
   * @returns A promise that resolves to an array of Project objects.
   * @throws {@link MemMachineAPIError} if the request fails.
   */
  async getProjects(): Promise<Project[]> {
    try {
      const response = await this.client.post('/api/v2/projects/list')
      return response.data
    } catch (error: unknown) {
      handleAPIError(error, 'Failed to get projects')
    }
  }

  /**
   * Checks the health status of the MemMachine server.
   *
   * @returns A promise that resolves to the server status information.
   * @throws {@link MemMachineAPIError} if the request fails.
   */
  async healthCheck(): Promise<HealthStatus> {
    try {
      const response = await this.client.get('/api/v2/health')
      return response.data
    } catch (error: unknown) {
      handleAPIError(error, 'Failed to check health status')
    }
  }
}

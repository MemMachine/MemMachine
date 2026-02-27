import type { AxiosInstance } from 'axios'

import { handleAPIError, MemMachineAPIError } from '@/errors'
import { MemMachineMemory, type MemoryContext } from '@/memory'
import type { CreateProjectOptions, Project, ProjectContext } from './memmachine-project.types'

/**
 * Provides methods to manage and interact with projects in MemMachine.
 *
 * @remarks
 * - Requires an AxiosInstance for making API requests.
 * - Provides access to MemMachineMemory instances for memory management within the project.
 * - Supports creating, retrieving, and deleting projects.
 *
 * Features:
 * - Access memory management using MemMachineMemory instances
 * - Create a new project using the provided project context and options
 * - Retrieve the project from MemMachine server
 * - Retrieve the count of episodic memories in the project
 * - Delete the project from MemMachine server
 *
 * @example
 * ```typescript
 * import MemMachineClient from '@memmachine/client'
 *
 * async function run() {
 *  const client = new MemMachineClient({ api_key: 'your_api_key' })
 *  const project = client.project({ org_id: 'your_org_id', project_id: 'your_project_id' })
 *
 *  // Create a new project on MemMachine server if it doesn't exist
 *  const newProject = await project.create({ description: 'New Project' })
 *  console.dir(newProject, { depth: null })
 *
 *  // Access memory management within the project
 *  const memory = project.memory()
 *  console.log(memory.getContext())
 *
 *  // Retrieve the project from MemMachine server
 *  const projectEntity = await project.get()
 *  console.dir(projectEntity, { depth: null })
 *
 *  // Retrieve episodic memory count
 *  const episodeCount = await project.getEpisodicCount()
 *  console.log(`Episodic Memory Count: ${episodeCount}`)
 *
 *  // Delete the project from MemMachine server
 *  await project.delete()
 * }
 *
 * run()
 * ```
 *
 * @param client - AxiosInstance for API communication.
 * @param projectContext - Options to configure the project context, see {@link ProjectContext}.
 */
export class MemMachineProject {
  client: AxiosInstance
  projectContext: ProjectContext

  constructor(client: AxiosInstance, projectContext: ProjectContext) {
    this.client = client

    const { org_id, project_id } = projectContext
    if (typeof org_id !== 'string' || !org_id.trim()) {
      throw new MemMachineAPIError('Organization ID must be a non-empty string')
    }
    if (typeof project_id !== 'string' || !project_id.trim()) {
      throw new MemMachineAPIError('Project ID must be a non-empty string')
    }

    this.projectContext = projectContext
  }

  /**
   * Creates a MemMachineMemory instance for managing memories within this project.
   *
   * @param memoryContext Context options for the memory.
   * @returns A MemMachineMemory instance.
   */
  memory(memoryContext?: MemoryContext): MemMachineMemory {
    return new MemMachineMemory(this.client, this.projectContext, memoryContext)
  }

  /**
   * Creates a new project in MemMachine.
   *
   * @param options Options for creating the project.
   * @returns A promise that resolves to the created Project.
   * @throws {@link MemMachineAPIError} if the API request fails.
   */
  async create(options?: CreateProjectOptions): Promise<Project> {
    const { description = '', reranker = '', embedder = '' } = options ?? {}

    const payload = {
      ...this.projectContext,
      description,
      config: {
        reranker,
        embedder
      }
    }

    try {
      const response = await this.client.post('/projects', payload)
      return response.data
    } catch (error: unknown) {
      handleAPIError(error, `Failed to create project with payload: ${JSON.stringify(payload)}`)
    }
  }

  /**
   * Retrieves the project from MemMachine.
   *
   * @returns A promise that resolves to the Project.
   * @throws {@link MemMachineAPIError} if the API request fails.
   */
  async get(): Promise<Project> {
    const payload = {
      ...this.projectContext
    }

    try {
      const response = await this.client.post('/projects/get', payload)
      return response.data
    } catch (error: unknown) {
      handleAPIError(error, `Failed to get project with payload: ${JSON.stringify(payload)}`)
    }
  }

  /**
   * Retrieves the count of episodic memories in the project.
   *
   * @return A promise that resolves to the count of episodic memories.
   * @throws {@link MemMachineAPIError} if the API request fails.
   */
  async getEpisodicCount(): Promise<number> {
    const payload = {
      ...this.projectContext
    }

    try {
      const response = await this.client.post('/projects/episode_count/get', payload)
      return response.data?.count ?? 0
    } catch (error: unknown) {
      handleAPIError(error, `Failed to get episodic memory count with payload: ${JSON.stringify(payload)}`)
    }
  }

  /**
   * Deletes the project from MemMachine.
   *
   * @returns A promise that resolves to null upon successful deletion.
   * @throws {@link MemMachineAPIError} if the API request fails.
   */
  async delete(): Promise<null> {
    const payload = {
      ...this.projectContext
    }

    try {
      const response = await this.client.post('/projects/delete', payload)
      return response.data
    } catch (error: unknown) {
      handleAPIError(error, `Failed to delete project with payload: ${JSON.stringify(payload)}`)
    }
  }
}

/**
 * Represents a project entity in MemMachine.
 *
 * @property org_id - The organization ID the project belongs to.
 * @property project_id - The unique identifier for the project.
 * @property description - A brief description of the project.
 * @property config - Project configuration, including reranker and embedder models.
 */
export interface Project {
  org_id: string
  project_id: string
  description?: string
  config?: {
    reranker: string
    embedder: string
  }
}

/**
 * Options for specifying a project context.
 *
 * @property org_id - The organization ID the project belongs to.
 * @property project_id - The unique identifier for the project.
 */
export interface ProjectContext {
  org_id: string
  project_id: string
}

/**
 * Options for creating a new project.
 *
 * @property description - A brief description of the project.
 * @property reranker - The reranker model to use.
 * @property embedder - The embedder model to use.
 */
export interface CreateProjectOptions {
  description?: string
  reranker?: string
  embedder?: string
}

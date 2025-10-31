/**
 * Options for initializing the MemoryClient.
 *
 * @property apiKey - API key for authentication (required)
 * @property host - Base URL of the MemMachine server (required)
 * @property timeout - Request timeout in milliseconds (optional)
 */
export interface ClientOptions {
  apiKey: string
  host: string
  timeout?: number
}

/**
 * Represents a memory session in MemMachine.
 *
 * @property session_id - Unique identifier for the session.
 * @property user_ids - Array of user IDs associated with the session.
 * @property group_id - Group ID associated with the session.
 * @property agent_ids - Array of agent IDs associated with the session.
 */
export interface MemorySession {
  session_id: string
  user_ids: string[]
  group_id: string
  agent_ids: string[]
}

/**
 * Options for specifying a memory session context.
 *
 * @property session_id - Session identifier (required)
 * @property user_id - Array of user IDs (optional)
 * @property group_id - Group ID (optional)
 * @property agent_id - Array of agent IDs (optional)
 */
export interface SessionOptions {
  session_id: string
  user_id?: string[]
  group_id?: string
  agent_id?: string[]
}

/**
 * Options for creating a memory in MemMachine.
 *
 * @property session_id - Session identifier (required, inherited from {@link SessionOptions})
 * @property user_id - Array of user IDs (optional, inherited from {@link SessionOptions})
 * @property group_id - Group ID (optional, inherited from {@link SessionOptions})
 * @property agent_id - Array of agent IDs (optional, inherited from {@link SessionOptions})
 * @property producer - Producer Entity ID (required)
 * @property produced_for - Target Entity ID (required)
 * @property metadata - Additional metadata (optional)
 */
export interface MemoryOptions extends SessionOptions {
  producer: string
  produced_for: string
  metadata?: Record<string, unknown>
}

/**
 * Options for searching memories in MemMachine.
 *
 * @property session_id - Session identifier (required, inherited from {@link SessionOptions})
 * @property user_id - Array of user IDs (optional, inherited from {@link SessionOptions})
 * @property group_id - Group ID (optional, inherited from {@link SessionOptions})
 * @property agent_id - Array of agent IDs (optional, inherited from {@link SessionOptions})
 * @property filter - Additional filter criteria (optional)
 * @property limit - Maximum number of results to return (optional)
 */
export interface SearchOptions extends SessionOptions {
  filter?: Record<string, unknown>
  limit?: number
}

/**
 * Represents the result of a memory search operation.
 *
 * @property status - Status code of the search operation result.
 * @property content - Search result content, typically containing results from both memory types.
 */
export interface SearchResult {
  status: number
  content: Record<string, any>
}

/**
 * Represents the health status of the MemMachine server.
 *
 * @property status - Overall health status (e.g., 'healthy').
 * @property service - Service name or identifier.
 * @property version - Server version string.
 * @property memory_managers - Object indicating the status of profile and episodic memory managers.
 *   - profile_memory: Whether the profile memory manager is healthy.
 *   - episodic_memory: Whether the episodic memory manager is healthy.
 */
export interface HealthStatus {
  status: string
  service: string
  version: string
  memory_managers: {
    profile_memory: boolean
    episodic_memory: boolean
  }
}

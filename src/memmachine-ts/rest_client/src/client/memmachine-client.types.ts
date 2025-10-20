/**
 * Options for initializing the MemMachine Client.
 *
 * @property base_url - Base URL of the MemMachine server (required)
 * @property api_key - API key for authentication (optional)
 * @property timeout - Request timeout in milliseconds (optional)
 * @property max_retries - Maximum number of retry attempts for failed requests (optional)
 */
export interface ClientOptions {
  base_url: string

  api_key?: string
  timeout?: number
  max_retries?: number
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
 * Type of memory session.
 *
 * Possible values:
 * - 'user' - User-specific memory session
 * - 'group' - Group-specific memory session
 * - 'agent' - Agent-specific memory session
 */
export type MemorySessionType = 'user' | 'group' | 'agent'

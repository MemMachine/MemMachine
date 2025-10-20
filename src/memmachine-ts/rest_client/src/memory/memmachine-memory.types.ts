/**
 * Types of memory available in MemMachine.
 *
 * Possible values:
 * - 'episodic' - Episodic memory type
 * - 'profile' - Profile memory type
 */
export type MemoryType = 'episodic' | 'profile'

/**
 * Represents the context of a memory session.
 *
 * @property session_id - Unique identifier for the session.
 * @property user_ids - Array of user IDs associated with the session.
 * @property group_id - Group ID associated with the session.
 * @property agent_ids - Array of agent IDs associated with the session.
 */
export interface MemoryContext {
  session_id: string
  user_ids: string[]
  group_id: string
  agent_ids: string[]
}

/**
 * Options for specifying a memory session context.
 *
 * @property session_id - Session identifier (optional)
 * @property user_id - Array of user ID(s) (optional)
 * @property group_id - Group ID (optional)
 * @property agent_id - Array of agent ID(s) (optional)
 */
export interface MemoryContextOptions {
  session_id?: string
  user_id?: string | string[]
  group_id?: string
  agent_id?: string | string[]
}

/**
 * Options for creating a memory in MemMachine.
 *
 * @property producer - Producer Entity ID (optional)
 * @property produced_for - Target Entity ID (optional)
 * @property episode_type - Type of episode (optional)
 * @property metadata - Additional metadata (optional)
 */
export interface AddMemoryOptions {
  producer?: string
  produced_for?: string
  episode_type?: string
  metadata?: Record<string, unknown>
}

/**
 * Options for searching memories in MemMachine.
 *
 * @property filter - Additional filter criteria (optional)
 * @property limit - Maximum number of results to return (optional)
 */
export interface SearchMemoryOptions {
  filter?: Record<string, unknown>
  limit?: number
}

/**
 * Represents the result of a memory search operation.
 *
 * @property status - Status code of the search operation result.
 * @property content - Search result content, typically containing results from both memory types.
 */
export interface SearchMemoryResult {
  status: number
  content: Record<string, any>
}

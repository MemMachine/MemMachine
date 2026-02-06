/**
 * Types of memory available in MemMachine.
 *
 * Possible values:
 * - 'episodic' - Episodic memory type
 * - 'semantic' - Semantic memory type
 */
export type MemoryType = 'episodic' | 'semantic'

/**
 * Roles that can produce memory entries.
 *
 * Possible values:
 * - 'user' - User role
 * - 'assistant' - Assistant role
 * - 'system' - System role
 */
export type MemoryProducerRole = 'user' | 'assistant' | 'system'

/**
 * Represents an episodic memory entry in MemMachine.
 *
 * @property uid - Unique identifier for the memory entry.
 * @property score - Relevance score of the memory entry.
 * @property content - Content of the memory entry.
 * @property created_at - Timestamp when the memory entry was created.
 * @property producer_id - ID of the entity that produced the memory entry.
 * @property producer_role - Role of the producer.
 * @property produced_for_id - ID of the entity for whom the memory was produced.
 * @property episode_type - Type of episode associated with the memory entry.
 * @property metadata - Additional metadata associated with the memory entry.
 */
export interface EpisodicMemory {
  uid: string
  score: number
  content: string
  created_at: string

  producer_id: string
  producer_role: string
  produced_for_id?: string

  episode_type: string
  metadata?: Record<string, unknown>
}

/**
 * Represents a semantic memory entry in MemMachine.
 *
 * @property set_id - Identifier for the memory set.
 * @property category - Category of the memory entry.
 * @property tag - Tag associated with the memory entry.
 * @property feature_name - Name of the feature.
 * @property value - Value of the memory entry.
 * @property metadata - Metadata associated with the memory entry, including citations, ID, and other information.
 */
export interface SemanticMemory {
  set_id: string
  category: string
  tag: string
  feature_name: string
  value: string
  metadata: {
    citations?: string[]
    id?: string
    other?: Record<string, unknown>
  }
}

/**
 * Options for specifying a memory context.
 *
 * @property session_id - Session identifier (optional).
 * @property user_id - User ID (optional).
 * @property group_id - Group ID (optional).
 * @property agent_id - Agent ID (optional).
 */
export interface MemoryContext {
  session_id?: string
  user_id?: string
  group_id?: string
  agent_id?: string
}

/**
 * Options for creating a memory in MemMachine.
 *
 * @property producer - Producer Entity ID (optional).
 * @property role - Role of the producer (optional).
 * @property produced_for - Target Entity ID (optional).
 * @property episode_type - Type of episode (optional).
 * @property timestamp - Timestamp of the memory entry (optional).
 * @property metadata - Additional metadata (optional).
 * @property types - Types of memory to create (optional).
 */
export interface AddMemoryOptions {
  producer?: string
  role?: MemoryProducerRole
  produced_for?: string
  episode_type?: string
  timestamp?: string
  metadata?: Record<string, string>
  types?: MemoryType[]
}

/**
 * Represents the result of adding memory to MemMachine.
 *
 * @property results - Array of results, each containing the unique identifier (uid) of the added memory entry.
 */
export interface AddMemoryResult {
  results: { uid: string }[]
}

/**
 * Options for searching memories in MemMachine.
 *
 * @property top_k - Maximum number of results to return (optional).
 * @property filter - Filter criteria for the search (optional).
 * @property expand_context - Number of extra episodes to include for context (optional).
 * @property score_threshold - Minimum score threshold for search results (optional).
 * @property types - Types of memory to search (optional).
 */
export interface SearchMemoriesOptions {
  top_k?: number
  filter?: string
  expand_context?: number
  score_threshold?: number
  types?: MemoryType[]
}

/**
 * Represents the result of searching memories in MemMachine.
 *
 * @property status - Status code of the search operation result.
 * @property content - Content of the search result, including episodic and semantic memories.
 */
export interface SearchMemoriesResult {
  status: number
  content: {
    episodic_memory: {
      long_term_memory: {
        episodes: EpisodicMemory[]
      }
      short_term_memory: {
        episodes: EpisodicMemory[]
        episode_summary: string[]
      }
    }
    semantic_memory: SemanticMemory[]
  }
}

/**
 * Options for listing memories in MemMachine.
 *
 * @property page_size - Number of memories per page (optional).
 * @property page_num - Page number to retrieve (optional).
 * @property filter - Filter criteria for listing memories (optional).
 * @property type - Type of memory to list (optional).
 */
export interface ListMemoriesOptions {
  page_size?: number
  page_num?: number
  filter?: string
  type?: MemoryType
}

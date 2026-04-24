/**
 * Main exports for the MemMachine memory module.
 *
 * @packageDocumentation
 * @module memmachine-memory
 */
export { MemMachineMemory } from './memmachine-memory'
export { formatEpisodes, formatSemanticMemories, formatSearchResult } from './format'
export type { FormattableEpisode } from './format'

export type {
  MemoryType,
  EpisodicMemory,
  ListEpisodicMemory,
  SemanticMemory,
  MemoryContext,
  MemoryProducerRole,
  AddMemoryOptions,
  AddMemoryResult,
  SearchMemoriesOptions,
  SearchMemoriesResult,
  ListMemoriesOptions,
  ListMemoriesResult,
} from './memmachine-memory.types'

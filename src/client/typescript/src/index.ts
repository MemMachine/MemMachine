/**
 * Entry point for the MemMachine TypeScript client.
 *
 * Provides the main API client (`MemoryClient`), error class (`APIError`), and all related types.
 */
import { MemoryClient } from './memmachine'

export { MemoryClient }
export default MemoryClient
export { APIError } from './api-error'

export type {
  ClientOptions,
  MemorySession,
  SessionOptions,
  MemoryOptions,
  SearchOptions,
  SearchResult,
  HealthStatus
} from './memmachine.types'

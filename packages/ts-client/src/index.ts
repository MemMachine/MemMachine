/**
 * Main entry point for the MemMachine TypeScript REST client library.
 * This module exports the main client, project, memory management classes, and error handling utilities.
 *
 * @packageDocumentation
 */
export * from '@/client'
export * from '@/project'
export * from '@/memory'

export { MemMachineAPIError } from '@/errors'

export { MemMachineClient as default } from '@/client'

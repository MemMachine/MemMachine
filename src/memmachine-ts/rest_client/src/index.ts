/**
 * Main entry point for the MemMachine TypeScript REST client library.
 * This module exports the primary classes, error classes, and memory management utilities.
 *
 * @packageDocumentation
 */
export * from '@/client'
export * from '@/memory'

export { MemMachineAPIError } from '@/errors/memmachine-api-error'

export { MemMachineClient as default } from '@/client'

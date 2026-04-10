import { AxiosError } from 'axios'

import { MemMachineAPIError } from './memmachine-api-error'

/**
 * Throws a MemMachineAPIError with a formatted message based on the provided error and context.
 *
 * @param error - The original error object or value to handle.
 * @param message - Contextual message describing the operation or error source.
 * @throws {MemMachineAPIError} Always throws a wrapped error with context information.
 */
export function handleAPIError(error: unknown, message: string): never {
  if (error instanceof AxiosError && error.response?.data?.detail) {
    throw new MemMachineAPIError(`${message}: ${error.message} - ${error.response.data.detail}`)
  }
  if (error instanceof Error) {
    throw new MemMachineAPIError(`${message}: ${error.message}`)
  }
  throw new MemMachineAPIError(`${message}: ${JSON.stringify(error)}`)
}

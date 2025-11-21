import { MemMachineAPIError } from './memmachine-api-error'

export function handleApiError(error: unknown, message: string): never {
  if (error instanceof Error) {
    throw new MemMachineAPIError(`${message}: ${error.message}`)
  }
  throw new MemMachineAPIError(`${message}: ${JSON.stringify(error)}`)
}

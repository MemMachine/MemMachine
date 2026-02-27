/**
 * Custom error class for MemMachine API errors.
 *
 * Used to represent errors returned by the MemMachine client methods.
 *
 * @extends Error
 * @param message - Error message describing the failure.
 */
export class MemMachineAPIError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'MemMachineAPIError'
  }
}

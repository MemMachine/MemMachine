import * as MemMachine from '@/index'

describe('Index Exports', () => {
  it('should export MemMachineClient', () => {
    expect(typeof MemMachine.MemMachineClient).toBe('function')
  })

  it('should export MemMachineAPIError', () => {
    expect(typeof MemMachine.MemMachineAPIError).toBe('function')
  })
})

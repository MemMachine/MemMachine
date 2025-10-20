import { MemMachineClient } from '@/client'

describe('MemMachine Client', () => {
  afterEach(() => {
    jest.restoreAllMocks()
  })

  it('should initialize MemMachine Client correctly', () => {
    const client = new MemMachineClient({ base_url: 'http://localhost:8080' })
    expect(client).toBeInstanceOf(MemMachineClient)
  })

  it('should throw error if base url is missing', () => {
    expect(() => {
      new MemMachineClient({ base_url: '' })
    }).toThrow('Base URL must be a non-empty string')
  })

  it('should retrieve sessions successfully', async () => {
    const client = new MemMachineClient({ base_url: 'http://localhost:8080' })
    jest.spyOn(client.client, 'get').mockResolvedValue({
      data: { sessions: [] }
    })

    const result = await client.getSessions()
    expect(result).toEqual([])
  })

  it('should handle error when retrieving sessions', async () => {
    const client = new MemMachineClient({ base_url: 'http://localhost:8080' })
    jest.spyOn(client.client, 'get').mockRejectedValue(new Error('Network Error'))

    await expect(client.getSessions()).rejects.toThrow(
      'Failed to fetch sessions from url: /v1/sessions/: Network Error'
    )
  })

  it('should perform health check successfully', async () => {
    const client = new MemMachineClient({ base_url: 'http://localhost:8080' })
    jest.spyOn(client.client, 'get').mockResolvedValue({
      data: { status: 'healthy' }
    })
    const result = await client.healthCheck()
    expect(result).toEqual({ status: 'healthy' })
  })

  it('should handle error when performing health check', async () => {
    const client = new MemMachineClient({ base_url: 'http://localhost:8080' })
    jest.spyOn(client.client, 'get').mockRejectedValue(new Error('Network Error'))

    await expect(client.healthCheck()).rejects.toThrow('Failed to check health status')
  })
})

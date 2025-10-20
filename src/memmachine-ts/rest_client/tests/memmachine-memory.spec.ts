import { MemMachineClient } from '@/client'

describe('MemMachine Memory', () => {
  afterEach(() => {
    jest.restoreAllMocks()
  })

  it('should initialize MemMachine Memory correctly', () => {
    const contextOptions = {
      session_id: 'test-session',
      user_id: 'test-user',
      agent_id: 'test-agent'
    }
    const client = new MemMachineClient({ base_url: 'http://localhost:8080' })
    const memory = client.memory(contextOptions)
    const context = memory.getMemoryContext()
    expect(context.session_id).toBe('test-session')
    expect(context.user_ids).toEqual(['test-user'])
    expect(context.group_id).toBe('test-user')
    expect(context.agent_ids).toEqual(['test-agent'])
  })

  it('should throw error if no user id is provided', () => {
    const client = new MemMachineClient({ base_url: 'http://localhost:8080' })
    expect(() => {
      client.memory({ agent_id: 'test-agent' })
    }).toThrow('At least one user id must be provided in MemoryContextOptions')
  })

  it('should add memory successfully', async () => {
    const client = new MemMachineClient({ base_url: 'http://localhost:8080' })
    jest.spyOn(client.client, 'post').mockResolvedValue({
      data: { status: 200 }
    })

    const memory = client.memory({ user_id: 'test-user' })
    const addResponse = await memory.add('Test memory content')
    expect(addResponse).toHaveProperty('status', 200)
  })

  it('should handle error when adding memory', async () => {
    const client = new MemMachineClient({ base_url: 'http://localhost:8080' })
    jest.spyOn(client.client, 'post').mockRejectedValue(new Error('Network Error'))
    const memory = client.memory({ user_id: 'test-user' })
    await expect(memory.add('Test memory content')).rejects.toThrow(
      /Failed to add memory with payload: .*: Network Error/
    )
  })

  it('should search memory successfully', async () => {
    const client = new MemMachineClient({ base_url: 'http://localhost:8080' })
    jest.spyOn(client.client, 'post').mockResolvedValue({
      data: { status: 0, content: {} }
    })
    const memory = client.memory({ user_id: 'test-user' })
    const searchResponse = await memory.search('Test query')
    expect(searchResponse).toEqual({ status: 0, content: {} })
  })

  it('should handle error when searching memory', async () => {
    const client = new MemMachineClient({ base_url: 'http://localhost:8080' })
    jest.spyOn(client.client, 'post').mockRejectedValue(new Error('Network Error'))
    const memory = client.memory({ user_id: 'test-user' })
    await expect(memory.search('Test query')).rejects.toThrow(
      /Failed to search memory with payload: .*: Network Error/
    )
  })
})

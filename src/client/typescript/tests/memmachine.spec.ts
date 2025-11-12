import { MemoryClient } from '@/memmachine'

describe('MemMachine Client', () => {
  afterEach(() => {
    jest.restoreAllMocks()
  })

  it('should initialize MemoryClient correctly', () => {
    const client = new MemoryClient({ apiKey: 'test-api-key', host: 'http://localhost:8080' })
    expect(client).toBeInstanceOf(MemoryClient)
    expect(client.apiKey).toBe('test-api-key')
    expect(client.host).toBe('http://localhost:8080')
  })

  it('should throw error if apiKey is missing', () => {
    expect(() => {
      new MemoryClient({ apiKey: '', host: 'http://localhost:8080' })
    }).toThrow('Memmachine API key must be a non-empty string')
  })

  it('should add memory successfully', async () => {
    const client = new MemoryClient({ apiKey: 'xxx', host: 'http://test' })
    jest.spyOn(client.client, 'post').mockResolvedValue({
      data: { status: 200 }
    })
    const result = await client.addMemory('Test memory content', 'note', {
      session_id: 'session_001',
      user_id: ['test_user'],
      producer: 'test_user',
      produced_for: 'test_user'
    })
    expect(result).toEqual({ status: 200 })
  })

  it('should handle error when adding memory', async () => {
    const client = new MemoryClient({ apiKey: 'xxx', host: 'http://test' })
    jest.spyOn(client.client, 'post').mockRejectedValue(new Error('Network Error'))

    await expect(
      client.addMemory('Test memory content', 'note', {
        session_id: 'session_001',
        user_id: ['test_user'],
        producer: 'test_user',
        produced_for: 'test_user'
      })
    ).rejects.toThrow(/Failed to add memory with payload: .*: Network Error/)
  })

  it('should throw error if memory content is empty', async () => {
    const client = new MemoryClient({ apiKey: 'xxx', host: 'http://test' })
    await expect(
      client.addMemory('', 'note', {
        session_id: 'session_001',
        user_id: ['test_user'],
        producer: 'test_user',
        produced_for: 'test_user'
      })
    ).rejects.toThrow('message field missing')
  })

  it('should add episodic memory successfully', async () => {
    const client = new MemoryClient({ apiKey: 'xxx', host: 'http://test' })
    jest.spyOn(client.client, 'post').mockResolvedValue({
      data: { status: 200 }
    })
    const result = await client.addEpisodicMemory('Episodic memory content', 'event', {
      session_id: 'session_002',
      user_id: ['test_user'],
      producer: 'test_user',
      produced_for: 'test_user'
    })
    expect(result).toEqual({ status: 200 })
  })

  it('should handle error when adding episodic memory', async () => {
    const client = new MemoryClient({ apiKey: 'xxx', host: 'http://test' })
    jest.spyOn(client.client, 'post').mockRejectedValue(new Error('Network Error'))

    await expect(
      client.addEpisodicMemory('Episodic memory content', 'event', {
        session_id: 'session_002',
        user_id: ['test_user'],
        producer: 'test_user',
        produced_for: 'test_user'
      })
    ).rejects.toThrow(/Failed to add episodic memory with payload: .*: Network Error/)
  })

  it('should add profile memory successfully', async () => {
    const client = new MemoryClient({ apiKey: 'xxx', host: 'http://test' })
    jest.spyOn(client.client, 'post').mockResolvedValue({
      data: { status: 200 }
    })
    const result = await client.addProfileMemory('Profile memory content', 'attribute', {
      session_id: 'session_003',
      user_id: ['test_user'],
      producer: 'test_user',
      produced_for: 'test_user'
    })
    expect(result).toEqual({ status: 200 })
  })

  it('should handle error when adding profile memory', async () => {
    const client = new MemoryClient({ apiKey: 'xxx', host: 'http://test' })
    jest.spyOn(client.client, 'post').mockRejectedValue(new Error('Network Error'))

    await expect(
      client.addProfileMemory('Profile memory content', 'attribute', {
        session_id: 'session_003',
        user_id: ['test_user'],
        producer: 'test_user',
        produced_for: 'test_user'
      })
    ).rejects.toThrow(/Failed to add profile memory with payload: .*: Network Error/)
  })

  it('should search memory successfully', async () => {
    const client = new MemoryClient({ apiKey: 'xxx', host: 'http://test' })
    jest.spyOn(client.client, 'post').mockResolvedValue({
      data: { status: 0, content: {} }
    })
    const result = await client.searchMemory('Test query', { session_id: 'session_001' })
    expect(result).toEqual({ status: 0, content: {} })
  })

  it('should handle error when searching memory', async () => {
    const client = new MemoryClient({ apiKey: 'xxx', host: 'http://test' })
    jest.spyOn(client.client, 'post').mockRejectedValue(new Error('Network Error'))

    await expect(client.searchMemory('Test query', { session_id: 'session_001' })).rejects.toThrow(
      /Failed to search memory with payload: .*: Network Error/
    )
  })

  it('should search episodic memory successfully', async () => {
    const client = new MemoryClient({ apiKey: 'xxx', host: 'http://test' })
    jest.spyOn(client.client, 'post').mockResolvedValue({
      data: { status: 0, content: {} }
    })
    const result = await client.searchEpisodicMemory('Episodic query', { session_id: 'session_002' })
    expect(result).toEqual({ status: 0, content: {} })
  })

  it('should handle error when searching episodic memory', async () => {
    const client = new MemoryClient({ apiKey: 'xxx', host: 'http://test' })
    jest.spyOn(client.client, 'post').mockRejectedValue(new Error('Network Error'))

    await expect(
      client.searchEpisodicMemory('Episodic query', { session_id: 'session_002' })
    ).rejects.toThrow(/Failed to search episodic memory with payload: .*: Network Error/)
  })

  it('should search profile memory successfully', async () => {
    const client = new MemoryClient({ apiKey: 'xxx', host: 'http://test' })
    jest.spyOn(client.client, 'post').mockResolvedValue({
      data: { status: 0, content: {} }
    })
    const result = await client.searchProfileMemory('Profile query', { session_id: 'session_003' })
    expect(result).toEqual({ status: 0, content: {} })
  })

  it('should handle error when searching profile memory', async () => {
    const client = new MemoryClient({ apiKey: 'xxx', host: 'http://test' })
    jest.spyOn(client.client, 'post').mockRejectedValue(new Error('Network Error'))

    await expect(client.searchProfileMemory('Profile query', { session_id: 'session_003' })).rejects.toThrow(
      /Failed to search profile memory with payload: .*: Network Error/
    )
  })

  it('should delete memory successfully', async () => {
    const client = new MemoryClient({ apiKey: 'xxx', host: 'http://test' })
    jest.spyOn(client.client, 'delete').mockResolvedValue({
      data: { status: 200 }
    })

    const result = await client.deleteMemory({ session_id: 'session_001' })
    expect(result).toEqual({ status: 200 })
  })

  it('should handle error when deleting memory', async () => {
    const client = new MemoryClient({ apiKey: 'xxx', host: 'http://test' })
    jest.spyOn(client.client, 'delete').mockRejectedValue(new Error('Network Error'))

    await expect(client.deleteMemory({ session_id: 'session_001' })).rejects.toThrow(
      /Failed to delete memory with payload: .*: Network Error/
    )
  })

  it('should retrieve sessions successfully', async () => {
    const client = new MemoryClient({ apiKey: 'xxx', host: 'http://test' })
    jest.spyOn(client.client, 'get').mockResolvedValue({
      data: { sessions: [] }
    })

    const result = await client.getSessions()
    expect(result).toEqual([])
  })

  it('should handle error when retrieving sessions', async () => {
    const client = new MemoryClient({ apiKey: 'xxx', host: 'http://test' })
    jest.spyOn(client.client, 'get').mockRejectedValue(new Error('Network Error'))

    await expect(client.getSessions()).rejects.toThrow(
      'Failed to fetch sessions from url: /v1/sessions/: Network Error'
    )
  })

  it('should retrieve user sessions successfully', async () => {
    const client = new MemoryClient({ apiKey: 'xxx', host: 'http://test' })
    jest.spyOn(client.client, 'get').mockResolvedValue({
      data: { sessions: [] }
    })
    const result = await client.getUserSessions('test_user')
    expect(result).toEqual([])
  })

  it('should handle error when retrieving user sessions', async () => {
    const client = new MemoryClient({ apiKey: 'xxx', host: 'http://test' })
    jest.spyOn(client.client, 'get').mockRejectedValue(new Error('Network Error'))

    await expect(client.getUserSessions('test_user')).rejects.toThrow(
      'Failed to fetch user sessions from url: /v1/users/test_user/sessions/: Network Error'
    )
  })

  it('should retrieve group sessions successfully', async () => {
    const client = new MemoryClient({ apiKey: 'xxx', host: 'http://test' })
    jest.spyOn(client.client, 'get').mockResolvedValue({
      data: { sessions: [] }
    })
    const result = await client.getGroupSessions('test_group')
    expect(result).toEqual([])
  })

  it('should handle error when retrieving group sessions', async () => {
    const client = new MemoryClient({ apiKey: 'xxx', host: 'http://test' })
    jest.spyOn(client.client, 'get').mockRejectedValue(new Error('Network Error'))

    await expect(client.getGroupSessions('test_group')).rejects.toThrow(
      'Failed to fetch group sessions from url: /v1/groups/test_group/sessions/: Network Error'
    )
  })

  it('should retrieve agent sessions successfully', async () => {
    const client = new MemoryClient({ apiKey: 'xxx', host: 'http://test' })
    jest.spyOn(client.client, 'get').mockResolvedValue({
      data: {
        sessions: []
      }
    })
    const result = await client.getAgentSessions('test_agent')
    expect(result).toEqual([])
  })

  it('should handle error when retrieving agent sessions', async () => {
    const client = new MemoryClient({ apiKey: 'xxx', host: 'http://test' })
    jest.spyOn(client.client, 'get').mockRejectedValue(new Error('Network Error'))

    await expect(client.getAgentSessions('test_agent')).rejects.toThrow(
      'Failed to fetch agent sessions from url: /v1/agents/test_agent/sessions/: Network Error'
    )
  })

  it('should perform health check successfully', async () => {
    const client = new MemoryClient({ apiKey: 'xxx', host: 'http://test' })
    jest.spyOn(client.client, 'get').mockResolvedValue({
      data: { status: 'healthy' }
    })
    const result = await client.healthCheck()
    expect(result).toEqual({ status: 'healthy' })
  })

  it('should handle error when performing health check', async () => {
    const client = new MemoryClient({ apiKey: 'xxx', host: 'http://test' })
    jest.spyOn(client.client, 'get').mockRejectedValue(new Error('Network Error'))

    await expect(client.healthCheck()).rejects.toThrow('Failed to check health status')
  })
})

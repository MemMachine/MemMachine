import { MemMachineClient } from '@/client'

const mockProjectContext = { org_id: 'test-org', project_id: 'test-project' }
const mockMemoryContext = { user_id: 'test-user', agent_id: 'test-agent' }

describe('MemMachine Memory', () => {
  afterEach(() => {
    jest.restoreAllMocks()
  })

  it('should initialize MemMachineMemory correctly', () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    const project = client.project(mockProjectContext)
    const memory = project.memory(mockMemoryContext)
    const context = memory.getContext()
    expect(context.org_id).toBe('test-org')
    expect(context.project_id).toBe('test-project')
    expect(context.user_id).toEqual('test-user')
    expect(context.agent_id).toEqual('test-agent')
  })

  it('should add memory successfully', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    jest.spyOn(client.client, 'post').mockResolvedValue({
      data: { results: [{ uid: '1' }] }
    })

    const project = client.project(mockProjectContext)
    const memory = project.memory(mockMemoryContext)
    const addResponse = await memory.add('Test memory content', { timestamp: new Date().toISOString() })
    expect(addResponse).toEqual({ results: [{ uid: '1' }] })
  })

  it('should throw error if role is invalid when adding memory', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    const project = client.project(mockProjectContext)
    const memory = project.memory(mockMemoryContext)
    // @ts-ignore
    await expect(memory.add('Test memory content', { role: 'invalid-role' })).rejects.toThrow(
      'Invalid memory role: invalid-role. Valid roles are: user, system, assistant'
    )
  })

  it('should handle error when adding memory', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    jest.spyOn(client.client, 'post').mockRejectedValue(new Error('Network Error'))
    const project = client.project(mockProjectContext)
    const memory = project.memory(mockMemoryContext)
    await expect(memory.add('Test memory content')).rejects.toThrow(
      /Failed to add memory with payload: .*: Network Error/
    )
  })

  it('should search memory successfully', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    jest.spyOn(client.client, 'post').mockResolvedValue({
      data: { status: 0, content: {} }
    })
    const project = client.project(mockProjectContext)
    const memory = project.memory(mockMemoryContext)
    const searchResponse = await memory.search('Test query')
    expect(searchResponse).toEqual({ status: 0, content: {} })
  })

  it('should throw error if query is empty when searching memory', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    const project = client.project(mockProjectContext)
    const memory = project.memory(mockMemoryContext)
    await expect(memory.search('')).rejects.toThrow('Search query must be a non-empty string')
  })

  it('should handle error when searching memory', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    jest.spyOn(client.client, 'post').mockRejectedValue(new Error('Network Error'))
    const project = client.project(mockProjectContext)
    const memory = project.memory(mockMemoryContext)
    await expect(memory.search('Test query')).rejects.toThrow(
      /Failed to search memories with payload: .*: Network Error/
    )
  })

  it('should list memories successfully', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    jest.spyOn(client.client, 'post').mockResolvedValue({
      data: { status: 0, content: {} }
    })
    const project = client.project(mockProjectContext)
    const memory = project.memory(mockMemoryContext)
    const listResponse = await memory.list({ page_size: 5, page_num: 0 })
    expect(listResponse).toEqual({ status: 0, content: {} })
  })

  it('should handle error when listing memories', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    jest.spyOn(client.client, 'post').mockRejectedValue(new Error('Network Error'))
    const project = client.project(mockProjectContext)
    const memory = project.memory(mockMemoryContext)
    await expect(memory.list({ page_size: 5, page_num: 0 })).rejects.toThrow(
      /Failed to list memories with payload: .*: Network Error/
    )
  })

  it('should delete episodic memory successfully', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    jest.spyOn(client.client, 'post').mockResolvedValue({
      data: null
    })
    const project = client.project(mockProjectContext)
    const memory = project.memory(mockMemoryContext)
    const deleteResponse = await memory.delete('1', 'episodic')
    expect(deleteResponse).toBeUndefined()
  })

  it('should delete semantic memory successfully', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    jest.spyOn(client.client, 'post').mockResolvedValue({
      data: null
    })
    const project = client.project(mockProjectContext)
    const memory = project.memory(mockMemoryContext)
    const deleteResponse = await memory.delete('1', 'semantic')
    expect(deleteResponse).toBeUndefined()
  })

  it('should throw error if id is empty when deleting memory', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    const project = client.project(mockProjectContext)
    const memory = project.memory(mockMemoryContext)
    await expect(memory.delete('', 'episodic')).rejects.toThrow('Memory ID must be a non-empty string')
  })

  it('should throw error if memory type is invalid when deleting memory', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    const project = client.project(mockProjectContext)
    const memory = project.memory(mockMemoryContext)
    // @ts-ignore
    await expect(memory.delete('1', 'invalid-type')).rejects.toThrow('Invalid memory type: invalid-type')
  })

  it('should handle error when deleting memory', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    jest.spyOn(client.client, 'post').mockRejectedValue(new Error('Network Error'))
    const project = client.project(mockProjectContext)
    const memory = project.memory(mockMemoryContext)
    await expect(memory.delete('1', 'episodic')).rejects.toThrow(
      /Failed to delete episodic memory with payload: .*: Network Error/
    )
  })
})

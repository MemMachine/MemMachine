import { MemMachineClient } from '@/client'

const mockProjectContext = { org_id: 'test-org', project_id: 'test-project' }
const mockMemoryContext = { user_id: 'test-user', agent_id: 'test-agent' }

describe('MemMachine Memory', () => {
  afterEach(() => {
    jest.restoreAllMocks()
  })

  it('should initialize MemMachineMemory correctly', () => {
    const client = new MemMachineClient({ base_url: 'http://localhost:8080' })
    const project = client.project(mockProjectContext)
    const memory = project.memory(mockMemoryContext)
    const context = memory.getContext()
    expect(context.org_id).toBe('test-org')
    expect(context.project_id).toBe('test-project')
    expect(context.user_id).toEqual('test-user')
    expect(context.agent_id).toEqual('test-agent')
  })

  it('should add memory successfully', async () => {
    const client = new MemMachineClient({ base_url: 'http://localhost:8080' })
    jest.spyOn(client.client, 'post').mockResolvedValue({
      data: { results: [{ uid: '1' }] }
    })

    const project = client.project(mockProjectContext)
    const memory = project.memory(mockMemoryContext)
    const addResponse = await memory.add('Test memory content')
    expect(addResponse).toEqual({ results: [{ uid: '1' }] })
  })

  it('should handle error when adding memory', async () => {
    const client = new MemMachineClient({ base_url: 'http://localhost:8080' })
    jest.spyOn(client.client, 'post').mockRejectedValue(new Error('Network Error'))
    const project = client.project(mockProjectContext)
    const memory = project.memory(mockMemoryContext)
    await expect(memory.add('Test memory content')).rejects.toThrow(
      /Failed to add memory with payload: .*: Network Error/
    )
  })

  it('should search memory successfully', async () => {
    const client = new MemMachineClient({ base_url: 'http://localhost:8080' })
    jest.spyOn(client.client, 'post').mockResolvedValue({
      data: { status: 0, content: {} }
    })
    const project = client.project(mockProjectContext)
    const memory = project.memory(mockMemoryContext)
    const searchResponse = await memory.search('Test query')
    expect(searchResponse).toEqual({ status: 0, content: {} })
  })

  it('should throw error if query is empty when searching memory', async () => {
    const client = new MemMachineClient({ base_url: 'http://localhost:8080' })
    const project = client.project(mockProjectContext)
    const memory = project.memory(mockMemoryContext)
    await expect(memory.search('')).rejects.toThrow('Search query must be a non-empty string')
  })

  it('should handle error when searching memory', async () => {
    const client = new MemMachineClient({ base_url: 'http://localhost:8080' })
    jest.spyOn(client.client, 'post').mockRejectedValue(new Error('Network Error'))
    const project = client.project(mockProjectContext)
    const memory = project.memory(mockMemoryContext)
    await expect(memory.search('Test query')).rejects.toThrow(
      /Failed to search memory with payload: .*: Network Error/
    )
  })
})

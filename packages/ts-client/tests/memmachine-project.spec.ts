import { MemMachineClient } from '@/client'

const mockProjectContext = { org_id: 'test-org', project_id: 'test-project' }
const mockProject = { org_id: 'test-org', project_id: 'test-project', description: 'Test Project' }

describe('MemMachine Project', () => {
  afterEach(() => {
    jest.restoreAllMocks()
  })

  it('should initialize MemMachineProject correctly', () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    const project = client.project(mockProjectContext)
    expect(project).toBeDefined()
    expect(project).toHaveProperty('projectContext')
    expect(project.projectContext).toEqual(mockProjectContext)
  })

  it('should throw error if org_id or project_id is missing', () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    expect(() => {
      client.project({ org_id: '', project_id: 'test-project' })
    }).toThrow('Organization ID must be a non-empty string')

    expect(() => {
      client.project({ org_id: 'test-org', project_id: '' })
    }).toThrow('Project ID must be a non-empty string')
  })

  it('should create project successfully', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    const project = client.project(mockProjectContext)
    jest.spyOn(project.client, 'post').mockResolvedValue({
      data: mockProject
    })
    const response = await project.create({ description: 'Test Project' })
    expect(response).toEqual(mockProject)
  })

  it('should handle error when creating project', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    const project = client.project(mockProjectContext)
    jest.spyOn(project.client, 'post').mockRejectedValue(new Error('Network Error'))
    await expect(project.create({ description: 'Test Project' })).rejects.toThrow(
      /Failed to create project with payload: .*: Network Error/
    )
  })

  it('should get project successfully', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    const project = client.project(mockProjectContext)
    jest.spyOn(project.client, 'post').mockResolvedValue({
      data: mockProject
    })
    const response = await project.get()
    expect(response).toEqual(mockProject)
  })

  it('should handle error when getting project', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    const project = client.project(mockProjectContext)
    jest.spyOn(project.client, 'post').mockRejectedValue(new Error('Network Error'))
    await expect(project.get()).rejects.toThrow(/Failed to get project with payload: .*: Network Error/)
  })

  it('should get episodic memory count successfully', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    const project = client.project(mockProjectContext)
    jest.spyOn(project.client, 'post').mockResolvedValue({
      data: { count: 42 }
    })
    const response = await project.getEpisodicCount()
    expect(response).toEqual(42)
  })

  it('should handle error when getting episodic memory count', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    const project = client.project(mockProjectContext)
    jest.spyOn(project.client, 'post').mockRejectedValue(new Error('Network Error'))
    await expect(project.getEpisodicCount()).rejects.toThrow(
      /Failed to get episodic memory count with payload: .*: Network Error/
    )
  })

  it('should delete project successfully', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    const project = client.project(mockProjectContext)
    jest.spyOn(project.client, 'post').mockResolvedValue({
      data: null
    })
    const response = await project.delete()
    expect(response).toBeNull()
  })

  it('should handle error when deleting project', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    const project = client.project(mockProjectContext)
    jest.spyOn(project.client, 'post').mockRejectedValue(new Error('Network Error'))
    await expect(project.delete()).rejects.toThrow(/Failed to delete project with payload: .*: Network Error/)
  })
})

import { MemMachineClient } from '@/client'

describe('MemMachine Client', () => {
  afterEach(() => {
    jest.restoreAllMocks()
  })

  it('should initialize MemMachine Client correctly', () => {
    const client = new MemMachineClient({
      base_url: 'http://localhost:8080/api/v2',
      timeout: 30000,
      max_retries: 2
    })
    expect(client).toBeInstanceOf(MemMachineClient)
  })

  it('should get projects successfully', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    const mockProjects = [
      { org_id: 'org-1', project_id: 'project-1' },
      { org_id: 'org-1', project_id: 'project-2' }
    ]
    jest.spyOn(client.client, 'post').mockResolvedValue({
      data: mockProjects
    })
    const projects = await client.getProjects()
    expect(projects).toEqual(mockProjects)
  })

  it('should handle error when getting projects', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    jest.spyOn(client.client, 'post').mockRejectedValue(new Error('Network Error'))

    await expect(client.getProjects()).rejects.toThrow('Failed to get projects')
  })

  it('should perform health check successfully', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    jest.spyOn(client.client, 'get').mockResolvedValue({
      data: { status: 'healthy' }
    })
    const result = await client.healthCheck()
    expect(result).toEqual({ status: 'healthy' })
  })

  it('should handle error when performing health check', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    jest.spyOn(client.client, 'get').mockRejectedValue(new Error('Network Error'))

    await expect(client.healthCheck()).rejects.toThrow('Failed to check health status')
  })
})

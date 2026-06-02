import { MemMachineClient } from '@/client'

const PROXY_ENV_VARS = ['HTTPS_PROXY', 'HTTP_PROXY', 'https_proxy', 'http_proxy'] as const

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

  it('should get metrics successfully', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    const mockMetrics = 'memmachine_requests_total 42'
    jest.spyOn(client.client, 'get').mockResolvedValue({
      data: mockMetrics
    })
    const metrics = await client.getMetrics()
    expect(metrics).toEqual(mockMetrics)
  })

  it('should handle error when getting metrics', async () => {
    const client = new MemMachineClient({ api_key: 'test-api-key' })
    jest.spyOn(client.client, 'get').mockRejectedValue(new Error('Network Error'))

    await expect(client.getMetrics()).rejects.toThrow('Failed to get metrics')
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

  describe('HTTP adapter configuration', () => {
    const savedEnv: Record<string, string | undefined> = {}

    beforeEach(() => {
      for (const key of PROXY_ENV_VARS) {
        savedEnv[key] = process.env[key]
        delete process.env[key]
      }
    })

    afterEach(() => {
      for (const key of PROXY_ENV_VARS) {
        if (savedEnv[key] === undefined) {
          delete process.env[key]
        } else {
          process.env[key] = savedEnv[key]
        }
      }
    })

    it('should forward an explicit adapter option to axios', () => {
      const client = new MemMachineClient({ adapter: 'fetch' })
      expect(client.client.defaults.adapter).toBe('fetch')
    })

    it('should auto-select the fetch adapter when HTTPS_PROXY is set', () => {
      process.env.HTTPS_PROXY = 'http://proxy.example:3128'
      const client = new MemMachineClient()
      expect(client.client.defaults.adapter).toBe('fetch')
    })

    it('should let an explicit adapter override a detected proxy env var', () => {
      process.env.HTTPS_PROXY = 'http://proxy.example:3128'
      const client = new MemMachineClient({ adapter: 'http' })
      expect(client.client.defaults.adapter).toBe('http')
    })

    it('should not force the fetch adapter when no proxy env var is present', () => {
      const client = new MemMachineClient()
      expect(client.client.defaults.adapter).not.toBe('fetch')
    })
  })
})

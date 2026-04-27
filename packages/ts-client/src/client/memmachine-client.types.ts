/**
 * Options for initializing the MemMachine Client.
 *
 * @property base_url - Base URL for the MemMachine server API (optional).
 * @property api_key - API key for authentication (optional).
 * @property timeout - Request timeout in milliseconds (optional).
 * @property max_retries - Maximum number of retry attempts for failed requests (optional).
 * @property adapter - Axios adapter to use for HTTP requests (optional).
 *   When unset, `'fetch'` is selected automatically if an HTTPS/HTTP proxy env var
 *   is detected (`HTTPS_PROXY`, `HTTP_PROXY`, `https_proxy`, or `http_proxy`);
 *   otherwise axios' default adapter is used.
 *   Set explicitly to opt in or out (e.g. `'fetch'` to force Node's native fetch,
 *   which handles CONNECT tunneling through TLS-terminating proxies correctly).
 */
export interface ClientOptions {
  base_url?: string
  api_key?: string
  timeout?: number
  max_retries?: number
  adapter?: 'http' | 'https' | 'fetch'
}

/**
 * Represents the health status of the MemMachine server.
 *
 * @property status - Overall health status (e.g., 'healthy').
 * @property service - Service name or identifier.
 * @property version - Server version string.
 * @property memory_managers - Object indicating the status of profile and episodic memory managers.
 *   - profile_memory: Whether the profile memory manager is healthy.
 *   - episodic_memory: Whether the episodic memory manager is healthy.
 */
export interface HealthStatus {
  status: string
  service: string
  version: string
  memory_managers: {
    profile_memory: boolean
    episodic_memory: boolean
  }
}

import * as https from 'https';
import * as http from 'http';
import { getApiBaseUrl, getAuthToken, API_ENDPOINTS } from './config';

export interface ApiResponse<T = any> {
    data: T;
    status: number;
    statusText: string;
}

export interface ApiError {
    message: string;
    status?: number;
    statusText?: string;
}

export class ApiClient {
    private baseUrl?: string;

    constructor(baseUrl?: string) {
        this.baseUrl = baseUrl; // Allow override for testing, but default to hot updates
    }

    /**
     * Generic HTTP request method
     */
    private async request<T = any>(
        endpoint: string,
        options: {
            method?: 'GET' | 'POST' | 'PUT' | 'DELETE';
            headers?: Record<string, string>;
            body?: any;
        } = {}
    ): Promise<ApiResponse<T>> {
        const { method = 'GET', headers = {}, body } = options;
        
        return new Promise((resolve, reject) => {
            // Always get the latest base URL from configuration for hot updates
            const currentBaseUrl = this.baseUrl ?? getApiBaseUrl();
            
            // Fix: Properly join the base URL path with the endpoint
            const baseUrlObj = new URL(currentBaseUrl + endpoint);

            const requestOptions = {
                hostname: baseUrlObj.hostname,
                port: baseUrlObj.port || (baseUrlObj.protocol === 'https:' ? 443 : 80),
                path: baseUrlObj.pathname + baseUrlObj.search,
                method,
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${getAuthToken()}`,
                    ...headers
                }
            };

            const protocol = baseUrlObj.protocol === 'https:' ? https : http;
            
            const req = protocol.request(requestOptions, (res) => {
                let data = '';
                
                res.on('data', (chunk) => {
                    data += chunk;
                });
                
                res.on('end', () => {
                    try {
                        const jsonData = data ? JSON.parse(data) : {};
                        resolve({
                            data: jsonData,
                            status: res.statusCode || 200,
                            statusText: res.statusMessage || 'OK'
                        });
                    } catch (error) {
                        reject({
                            message: 'Failed to parse response: ' + data,
                            status: res.statusCode,
                            statusText: res.statusMessage
                        });
                    }
                });
            });

            req.on('error', (error) => {
                reject({
                    message: 'Request failed: ' + error.message
                });
            });

            if (body) {
                req.write(JSON.stringify(body));
            }

            req.end();
        });
    }

    /**
     * GET request
     */
    async get<T = any>(endpoint: string, headers?: Record<string, string>): Promise<ApiResponse<T>> {
        return this.request<T>(endpoint, { method: 'GET', headers });
    }

    /**
     * POST request
     */
    async post<T = any>(endpoint: string, body?: any, headers?: Record<string, string>): Promise<ApiResponse<T>> {
        return this.request<T>(endpoint, { method: 'POST', headers, body });
    }

    /**
     * PUT request
     */
    async put<T = any>(endpoint: string, body?: any, headers?: Record<string, string>): Promise<ApiResponse<T>> {
        return this.request<T>(endpoint, { method: 'PUT', headers, body });
    }

    /**
     * DELETE request
     */
    async delete<T = any>(endpoint: string, headers?: Record<string, string>): Promise<ApiResponse<T>> {
        return this.request<T>(endpoint, { method: 'DELETE', headers });
    }

    /**
     * Build URL with query parameters
     */
    private buildUrlWithParams(endpoint: string, params?: Record<string, any>): string {
        if (!params || Object.keys(params).length === 0) {
            return endpoint;
        }
        
        const queryParams = new URLSearchParams();
        for (const [key, value] of Object.entries(params)) {
            if (value !== undefined && value !== null) {
                queryParams.append(key, String(value));
            }
        }
        
        const queryString = queryParams.toString();
        return queryString ? `${endpoint}?${queryString}` : endpoint;
    }

    // Specific API methods
    async getEpisodicMemory(params?: Record<string, any>): Promise<ApiResponse> {
        const url = this.buildUrlWithParams(API_ENDPOINTS.EPISODIC_MEMORY, params);
        console.log('getEpisodicMemory url:', url);
        return this.get(url);
    }

    async getProfileMemory(params?: Record<string, any>): Promise<ApiResponse> {
        const url = this.buildUrlWithParams(API_ENDPOINTS.PROFILE_MEMORY, params);
        return this.get(url);
    }

    async healthCheck(): Promise<ApiResponse> {
        return this.get(API_ENDPOINTS.HEALTH);
    }

    async getDebugInfo(): Promise<ApiResponse> {
        return this.get(API_ENDPOINTS.DEBUG);
    }

    async login(credentials: { username: string; password: string }): Promise<ApiResponse> {
        return this.post(API_ENDPOINTS.LOGIN, credentials);
    }

    async logout(): Promise<ApiResponse> {
        return this.post(API_ENDPOINTS.LOGOUT);
    }

    async getSessions(): Promise<ApiResponse> {
        return this.get(API_ENDPOINTS.SESSIONS);
    }

    async deleteMemory(): Promise<ApiResponse> {
        return this.delete(API_ENDPOINTS.DELETE_MEMORY);
    }
}

// Export a default instance
export const apiClient = new ApiClient();

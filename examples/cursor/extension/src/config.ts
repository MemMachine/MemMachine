import * as vscode from 'vscode';

// API Endpoints
export const API_ENDPOINTS = {
    EPISODIC_MEMORY: '/memory/get_episodic_memory',
    PROFILE_MEMORY: '/memory/get_profile_memory',
    DELETE_MEMORY: '/memory/delete',
    SESSIONS: '/sessions',
    HEALTH: '/health',
    DEBUG: '/debug',
    LOGIN: '/auth/login',
    LOGOUT: '/auth/logout'
} as const;



// Extension Settings
// export function getMcpUrl(): string {
//     return vscode.workspace.getConfiguration('memmachine').get('mcpUrl', 'http://ec2-18-223-182-61.us-east-2.compute.amazonaws.com:8001/mcp');
// }

export function getApiBaseUrl(): string {
    return vscode.workspace.getConfiguration('memmachine').get('apiBaseUrl', 'http://ec2-18-223-182-61.us-east-2.compute.amazonaws.com:8001/api');
}

export function getAuthToken(): string {
    return vscode.workspace.getConfiguration('memmachine').get('authToken', 'your-auth-token-here');
}


// Constants
export const MCP_NAME = 'MemMachine';
export const SESSION_PREFIX = 'PROJECT-';

export const REFRESH_INTERVAL = 1000 * 30; // 30 seconds
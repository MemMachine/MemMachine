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
export function getMcpUrl(): string {
    return vscode.workspace.getConfiguration('memmachine').get('mcpUrl', 'http://127.0.0.1:8001/mcp/');
}

export function getApiBaseUrl(): string {
    return vscode.workspace.getConfiguration('memmachine').get('apiBaseUrl', 'http://127.0.0.1:8001');
}

export function getAuthToken(): string {
    return vscode.workspace.getConfiguration('memmachine').get('authToken', '**your-auth-token-here**');
}

export const MCP_URL = getMcpUrl();
export const API_BASE_URL = getApiBaseUrl();
export const AUTH_TOKEN = getAuthToken();



// Constants
export const MCP_NAME = 'MemMachine';
export const SESSION_PREFIX = 'CURSOR-PROJECT-';

export const REFRESH_INTERVAL = 1000 * 30; // 30 seconds
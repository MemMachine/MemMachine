import * as vscode from 'vscode';
import { MCP_NAME, MCP_URL, AUTH_TOKEN } from './config';

export interface MCPServerConfig {
    name: string;
    url: string;
    headers?: Record<string, string>;
}

export class MCPServerManager {
    private static instance: MCPServerManager;
    private registeredServers: Map<string, MCPServerConfig> = new Map();
    private isCursor: boolean;

    private constructor() {
        // Detect if we're running in Cursor or VSCode
        this.isCursor = this.detectCursor();
    }

    public static getInstance(): MCPServerManager {
        if (!MCPServerManager.instance) {
            MCPServerManager.instance = new MCPServerManager();
        }
        return MCPServerManager.instance;
    }

    private detectCursor(): boolean {
        // Check if we're running in Cursor by looking for Cursor-specific APIs
        try {
            // @ts-ignore - Cursor-specific API
            return typeof vscode.cursor !== 'undefined' && typeof vscode.cursor.mcp !== 'undefined';
        } catch {
            return false;
        }
    }

    public async registerServer(config: MCPServerConfig): Promise<boolean> {
        try {
            if (this.isCursor) {
                return await this.registerServerInCursor(config);
            } else {
                return await this.registerServerInVSCode(config);
            }
        } catch (error) {
            console.error('Failed to register MCP server:', error);
            vscode.window.showErrorMessage(`Failed to register MCP server: ${error}`);
            return false;
        }
    }

    public async unregisterServer(serverName: string): Promise<boolean> {
        try {
            if (this.isCursor) {
                return await this.unregisterServerInCursor(serverName);
            } else {
                return await this.unregisterServerInVSCode(serverName);
            }
        } catch (error) {
            console.error('Failed to unregister MCP server:', error);
            vscode.window.showErrorMessage(`Failed to unregister MCP server: ${error}`);
            return false;
        }
    }

    private async registerServerInCursor(config: MCPServerConfig): Promise<boolean> {
        try {
            // @ts-ignore - Cursor-specific API
            await vscode.cursor.mcp.registerServer({
                name: config.name,
                server: {
                    url: config.url,
                    headers: config.headers || {}
                }
            });
            
            this.registeredServers.set(config.name, config);
            vscode.window.showInformationMessage(`MCP server '${config.name}' registered successfully in Cursor`);
            return true;
        } catch (error) {
            console.error('Cursor MCP registration failed:', error);
            throw error;
        }
    }

    private async registerServerInVSCode(config: MCPServerConfig): Promise<boolean> {
        try {
            // For VSCode, we'll use the standard MCP API if available
            // @ts-ignore - VSCode MCP API
            if (typeof vscode.mcp !== 'undefined' && typeof vscode.mcp.registerServer !== 'undefined') {
                // @ts-ignore - VSCode MCP API
                await vscode.mcp.registerServer({
                    name: config.name,
                    server: {
                        command: 'curl',
                        args: ['-X', 'POST', config.url],
                        env: config.headers || {}
                    }
                });
            } else {
                // Fallback: Write to VSCode settings
                await this.writeToVSCodeSettings(config);
            }
            
            this.registeredServers.set(config.name, config);
            vscode.window.showInformationMessage(`MCP server '${config.name}' registered successfully in VSCode`);
            return true;
        } catch (error) {
            console.error('VSCode MCP registration failed:', error);
            throw error;
        }
    }

    private async unregisterServerInCursor(serverName: string): Promise<boolean> {
        try {
            // @ts-ignore - Cursor-specific API
            await vscode.cursor.mcp.unregisterServer(serverName);
            
            this.registeredServers.delete(serverName);
            vscode.window.showInformationMessage(`MCP server '${serverName}' unregistered successfully from Cursor`);
            return true;
        } catch (error) {
            console.error('Cursor MCP unregistration failed:', error);
            throw error;
        }
    }

    private async unregisterServerInVSCode(serverName: string): Promise<boolean> {
        try {
            // @ts-ignore - VSCode MCP API
            if (typeof vscode.mcp !== 'undefined' && typeof vscode.mcp.unregisterServer !== 'undefined') {
                // @ts-ignore - VSCode MCP API
                await vscode.mcp.unregisterServer(serverName);
            } else {
                // Fallback: Remove from VSCode settings
                await this.removeFromVSCodeSettings(serverName);
            }
            
            this.registeredServers.delete(serverName);
            vscode.window.showInformationMessage(`MCP server '${serverName}' unregistered successfully from VSCode`);
            return true;
        } catch (error) {
            console.error('VSCode MCP unregistration failed:', error);
            throw error;
        }
    }

    private async writeToVSCodeSettings(config: MCPServerConfig): Promise<void> {
        const configSection = vscode.workspace.getConfiguration('mcp');
        const servers = configSection.get<Record<string, any>>('servers', {});
        
        servers[config.name] = {
            type: 'http',
            url: config.url,
            headers: config.headers || {}
        };
        
        await configSection.update('servers', servers, vscode.ConfigurationTarget.Global);
    }

    private async removeFromVSCodeSettings(serverName: string): Promise<void> {
        const configSection = vscode.workspace.getConfiguration('mcp');
        const servers = configSection.get<Record<string, any>>('servers', {});
        
        delete servers[serverName];
        
        await configSection.update('servers', servers, vscode.ConfigurationTarget.Global);
    }

    public getRegisteredServers(): MCPServerConfig[] {
        return Array.from(this.registeredServers.values());
    }

    public isServerRegistered(serverName: string): boolean {
        return this.registeredServers.has(serverName);
    }

    public getEnvironment(): 'cursor' | 'vscode' {
        return this.isCursor ? 'cursor' : 'vscode';
    }

    public async registerMemMachineServer(): Promise<boolean> {
        const MCPConfig: MCPServerConfig = {
            name: MCP_NAME,
            url: MCP_URL,
            headers: {
                'Authorization': `Bearer ${AUTH_TOKEN}`
            }
        };
        return await this.registerServer(MCPConfig);
    }
}

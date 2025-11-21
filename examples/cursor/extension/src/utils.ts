import * as vscode from 'vscode';
import * as path from 'path';
import { SESSION_PREFIX } from './config';

/**
 * Get the project name from the workspace
 * @returns The name of the current workspace folder, or undefined if no workspace is open
 */
export function getProjectName(): string | undefined {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (workspaceFolders && workspaceFolders.length > 0) {
        // Get the name of the first workspace folder
        return workspaceFolders[0].name;
    }
    return undefined;
}

/**
 * Get the project name with prefix
 * @returns The name of the current workspace folder with workspace prefix, or undefined if no workspace is open
 */
export function getProjectNameWithPrefix(): string | undefined {
    const projectName = getProjectName();
    if (projectName) {
        return SESSION_PREFIX + projectName;
    }
    return undefined;
}

/**
 * Get the project name from the workspace folder path
 * @returns The folder name extracted from the path, or undefined if no workspace is open
 */
export function getProjectNameFromPath(): string | undefined {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (workspaceFolders && workspaceFolders.length > 0) {
        // Get the folder name from the URI path
        return path.basename(workspaceFolders[0].uri.fsPath);
    }
    return undefined;
}

/**
 * Get the workspace name (for multi-root workspaces)
 * @returns The workspace name if defined in .code-workspace file, otherwise undefined
 */
export function getWorkspaceName(): string | undefined {
    return vscode.workspace.name;
}

/**
 * Get the workspace folder path
 * @returns The absolute path to the workspace folder, or undefined if no workspace is open
 */
export function getWorkspacePath(): string | undefined {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (workspaceFolders && workspaceFolders.length > 0) {
        return workspaceFolders[0].uri.fsPath;
    }
    return undefined;
}

/**
 * Get all workspace folders
 * @returns Array of workspace folder information, or empty array if no workspace is open
 */
export function getAllWorkspaceFolders(): Array<{ name: string; path: string }> {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders) {
        return [];
    }
    
    return workspaceFolders.map(folder => ({
        name: folder.name,
        path: folder.uri.fsPath
    }));
}


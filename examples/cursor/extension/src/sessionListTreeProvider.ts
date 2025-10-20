import * as vscode from 'vscode';
import { apiClient } from './apiClient';

export class SessionListTreeProvider implements vscode.TreeDataProvider<SessionItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<SessionItem | undefined | null | void> = new vscode.EventEmitter<SessionItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<SessionItem | undefined | null | void> = this._onDidChangeTreeData.event;

    private _sessions: SessionItem[] = [];
    private _isLoading: boolean = false;

    constructor() {
        this.refresh();
    }

    refresh(): void {
        this.loadSessions();
    }

    getTreeItem(element: SessionItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: SessionItem): Thenable<SessionItem[]> {
        if (!element) {
            const items = [...this._sessions];
            // Show loading indicator if refreshing
            if (this._isLoading) {
                items.unshift(new SessionItem('Loading...', 'Refreshing sessions...', vscode.TreeItemCollapsibleState.None, 'loading'));
            }
            return Promise.resolve(items);
        }
        
        // If element is a session, return its details
        if (element.sessionId && element.sessionId !== 'loading') {
            return Promise.resolve(this.createSessionDetailItems(element));
        }
        
        return Promise.resolve([]);
    }

    private createSessionDetailItems(sessionItem: SessionItem): SessionItem[] {
        const details: SessionItem[] = [];
        
        // Add session ID detail
        if (sessionItem.sessionId) {
            details.push(new SessionItem(
                'Session ID',
                sessionItem.sessionId,
                vscode.TreeItemCollapsibleState.None,
                `${sessionItem.sessionId}-id`,
                true
            ));
        }

        
        // Add group ID if available
        if (sessionItem.groupId) {
            details.push(new SessionItem(
                'Group',
                sessionItem.groupId,
                vscode.TreeItemCollapsibleState.None,
                `${sessionItem.sessionId}-group`,
                true
            ));
        }
        
        // Add agent IDs if available
        if (sessionItem.agentIds && sessionItem.agentIds.length > 0) {
            details.push(new SessionItem(
                'Agents',
                sessionItem.agentIds.join(', '),
                vscode.TreeItemCollapsibleState.None,
                `${sessionItem.sessionId}-agents`,
                true
            ));
        }
        
        return details;
    }

    private async loadSessions(): Promise<void> {
        this._isLoading = true;
        this._onDidChangeTreeData.fire();
        
        try {
            const response = await apiClient.getSessions();
            this._sessions = this.parseSessions(response.data);
        } catch (error) {
            console.error('Failed to load sessions:', error);
            this._sessions = [new SessionItem('Error loading sessions', 'error', vscode.TreeItemCollapsibleState.None)];
        } finally {
            this._isLoading = false;
            this._onDidChangeTreeData.fire();
        }
    }

    private parseSessions(data: any): SessionItem[] {
        console.log('Raw Sessions API data:', JSON.stringify(data, null, 2));
        
        if (!data) {
            return [new SessionItem('No sessions found', 'empty', vscode.TreeItemCollapsibleState.None)];
        }

        let sessions: any[] = [];
        
        // Handle different data structures
        if (Array.isArray(data)) {
            sessions = data;
        } else if (data.sessions && Array.isArray(data.sessions)) {
            sessions = data.sessions;
        } else if (data.data && Array.isArray(data.data)) {
            sessions = data.data;
        } else if (data.results && Array.isArray(data.results)) {
            sessions = data.results;
        } else {
            // If it's an object with session data, try to extract it
            sessions = Object.values(data).filter(item => 
                typeof item === 'object' && item !== null
            );
        }

        console.log('Parsed sessions:', sessions);

        if (sessions.length === 0) {
            return [new SessionItem('No sessions found', 'empty', vscode.TreeItemCollapsibleState.None)];
        }

        // Create session items
        return sessions.map((session, index) => {
            const sessionId = session.session_id || session.sessionId || `session-${index}`;
            const userIds = session.user_ids || session.userIds || [];
            const groupId = session.group_id || session.groupId || '';
            const agentIds = session.agent_ids || session.agentIds || [];
            
            // Create a display label for the session
            const label = `Session: ${sessionId}`;
            const description = userIds.length > 0 ? `${userIds.length} user(s)` : '';
            
            return new SessionItem(
                label,
                description,
                vscode.TreeItemCollapsibleState.Collapsed,
                sessionId,
                false,
                'session',
                userIds,
                groupId,
                agentIds
            );
        });
    }
}

export class SessionItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly content: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly sessionId?: string,
        public readonly isDetail: boolean = false,
        public readonly contextValue: string = 'session',
        public readonly userIds?: string[],
        public readonly groupId?: string,
        public readonly agentIds?: string[]
    ) {
        super(label, collapsibleState);
        
        // Set tooltip to show full content
        try {
            // Ensure content is a string and not null/undefined
            const safeContent = content && typeof content === 'string' ? content : String(content || '');
            this.tooltip = new vscode.MarkdownString(safeContent);
        } catch (error) {
            // Fallback to plain text if MarkdownString fails
            this.tooltip = content && typeof content === 'string' ? content : String(content || '');
        }
        
        // Set description to show a preview
        if (content && content !== 'empty' && content !== 'error' && typeof content === 'string') {
            this.description = content.length > 50 ? content.substring(0, 50) + '...' : content;
        } else {
            this.description = content && typeof content === 'string' ? content : String(content || '');
        }
        
        this.contextValue = contextValue;
        
        // Add icon based on item type
        if (this.sessionId === 'loading') {
            this.iconPath = new vscode.ThemeIcon('loading~spin');
        } else if (this.isDetail) {
            // Detail items show info icon
            this.iconPath = new vscode.ThemeIcon('info');
        } else if (contextValue === 'session') {
            // Session items show history icon
            this.iconPath = new vscode.ThemeIcon('history');
        }
    }
}


import * as vscode from 'vscode';
import { apiClient } from './apiClient';
import { getProjectName, getProjectNameWithPrefix } from './utils';

export class EpisodicMemoryTreeProvider implements vscode.TreeDataProvider<EpisodicMemoryItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<EpisodicMemoryItem | undefined | null | void> = new vscode.EventEmitter<EpisodicMemoryItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<EpisodicMemoryItem | undefined | null | void> = this._onDidChangeTreeData.event;

    private _memories: EpisodicMemoryItem[] = [];
    private _isLoading: boolean = false;

    constructor() {
        this.refresh();
    }

    refresh(): void {
        this.loadMemories();
    }

    async clear(): Promise<void> {
        // Show confirmation dialog
        const result = await vscode.window.showWarningMessage(
            'Are you sure you want to clear all episodic memories? This action cannot be undone.',
            { modal: true },
            'Clear'
        );

        if (result !== 'Clear') {
            return;
        }

        try {
            await apiClient.deleteMemory();
            vscode.window.showInformationMessage('Episodic memories cleared successfully');
            this.refresh();
        } catch (error) {
            console.error('Failed to clear episodic memories:', error);
            vscode.window.showErrorMessage('Failed to clear episodic memories');
        }
    }

    getTreeItem(element: EpisodicMemoryItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: EpisodicMemoryItem): Thenable<EpisodicMemoryItem[]> {
        if (!element) {
            const items = [...this._memories];
            // Show loading indicator if refreshing
            if (this._isLoading) {
                items.unshift(new EpisodicMemoryItem('Loading...', 'Refreshing episodic memories...', vscode.TreeItemCollapsibleState.None, 'loading'));
            }
            return Promise.resolve(items);
        }
        
        // If element is a memory item (not loading), return its details
        if (element.memoryId && element.memoryId !== 'loading') {
            return Promise.resolve(this.createMemoryDetailItems(element));
        }
        
        return Promise.resolve([]);
    }

    private createMemoryDetailItems(memoryItem: EpisodicMemoryItem): EpisodicMemoryItem[] {
        const details: EpisodicMemoryItem[] = [];
        
        // Add content detail
        if (memoryItem.content && memoryItem.content !== 'empty' && memoryItem.content !== 'error') {
            details.push(new EpisodicMemoryItem(
                'Content',
                memoryItem.content,
                vscode.TreeItemCollapsibleState.None,
                `${memoryItem.memoryId}-content`,
                true
            ));
        }
        
        // Add metadata details if available
        if (memoryItem.memoryId) {
            details.push(new EpisodicMemoryItem(
                'ID',
                memoryItem.memoryId,
                vscode.TreeItemCollapsibleState.None,
                `${memoryItem.memoryId}-id`,
                true
            ));
        }
        
        // Add timestamp if available (you can extend this based on your API response)
        details.push(new EpisodicMemoryItem(
            'Timestamp',
            new Date().toLocaleString(),
            vscode.TreeItemCollapsibleState.None,
            `${memoryItem.memoryId}-timestamp`,
            true
        ));
        
        return details;
    }

    private async loadMemories(): Promise<void> {
        this._isLoading = true;
        this._onDidChangeTreeData.fire();
        
        try {
            const response = await apiClient.getEpisodicMemory({
                projectId: getProjectNameWithPrefix()
            });
            this._memories = this.parseMemories(response.data);
        } catch (error) {
            console.error('Failed to load episodic memories:', error);
            this._memories = [new EpisodicMemoryItem('Error loading memories', 'error', vscode.TreeItemCollapsibleState.None)];
        } finally {
            this._isLoading = false;
            this._onDidChangeTreeData.fire();
        }
    }

    private parseMemories(data: any): EpisodicMemoryItem[] {
        console.log('Raw API data:', JSON.stringify(data, null, 2));
        
        if (!data) {
            return [new EpisodicMemoryItem('No memories found', 'empty', vscode.TreeItemCollapsibleState.None)];
        }

        let memories: any[] = [];
        
        // Handle different data structures
        if (Array.isArray(data)) {
            memories = data;
        } else if (data.memories && Array.isArray(data.memories)) {
            memories = data.memories;
        } else if (data.episodic_memory && Array.isArray(data.episodic_memory)) {
            memories = data.episodic_memory;
        } else if (data.data && Array.isArray(data.data)) {
            memories = data.data;
        } else if (data.results && Array.isArray(data.results)) {
            memories = data.results;
        } else {
            // If it's an object with memory data, try to extract it
            memories = Object.values(data).filter(item => 
                typeof item === 'object' && item !== null
            );
        }

        console.log('Parsed memories:', memories);

        if (memories.length === 0) {
            return [new EpisodicMemoryItem('No memories found', 'empty', vscode.TreeItemCollapsibleState.None)];
        }

        return memories.map((memory, index) => {
            // Try multiple possible field names for title
            const title = memory.title || 
                         memory.name || 
                         memory.subject || 
                         memory.heading || 
                         memory.label ||
                         `Memory ${index + 1}`;
            
            // Try multiple possible field names for content
            const content = memory.content || 
                           memory.description || 
                           memory.text || 
                           memory.body || 
                           memory.message ||
                           memory.details ||
                           memory.summary ||
                           JSON.stringify(memory, null, 2);
            
            const id = memory.id || memory.uuid || memory.key || index.toString();
            
            console.log(`Memory ${index + 1}:`, { title, content: content.substring(0, 100) + '...', id });
            
            return new EpisodicMemoryItem(title, content, vscode.TreeItemCollapsibleState.Collapsed, id);
        });
    }
}

export class EpisodicMemoryItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly content: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly memoryId?: string,
        public readonly isDetail: boolean = false
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
        
        this.contextValue = 'episodicMemory';
        
        // Add command based on item type
        if (this.memoryId === 'loading') {
            this.iconPath = new vscode.ThemeIcon('loading');
        } else if (this.isDetail) {
            // Detail items show info icon, no command needed
            // this.iconPath = new vscode.ThemeIcon('info');
        } else {
            // Main memory items are collapsible, no command needed
            this.iconPath = new vscode.ThemeIcon('book');
        }
    }

}

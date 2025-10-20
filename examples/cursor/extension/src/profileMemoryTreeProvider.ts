import * as vscode from "vscode";
import { apiClient } from "./apiClient";

export class ProfileMemoryTreeProvider
  implements vscode.TreeDataProvider<ProfileMemoryItem>
{
  private _onDidChangeTreeData: vscode.EventEmitter<
    ProfileMemoryItem | undefined | null | void
  > = new vscode.EventEmitter<ProfileMemoryItem | undefined | null | void>();
  readonly onDidChangeTreeData: vscode.Event<
    ProfileMemoryItem | undefined | null | void
  > = this._onDidChangeTreeData.event;

  private _memories: ProfileMemoryItem[] = [];
  private _isLoading: boolean = false;

  constructor() {
    this.refresh();
  }

  refresh(): void {
    this.loadMemories();
  }

  getTreeItem(element: ProfileMemoryItem): vscode.TreeItem {
    return element;
  }

  getChildren(element?: ProfileMemoryItem): Thenable<ProfileMemoryItem[]> {
    if (!element) {
      const items = [...this._memories];
      // Show loading indicator if refreshing
      if (this._isLoading) {
        items.unshift(
          new ProfileMemoryItem(
            "Loading...",
            "Refreshing profile memories...",
            vscode.TreeItemCollapsibleState.None,
            "loading"
          )
        );
      }
      return Promise.resolve(items);
    }

    // If element is a tag group, return the memory items in that tag
    if (element.memoryId && String(element.memoryId).startsWith("tag-")) {
      const tagMemories = (element as any).tagMemories;
      if (tagMemories && Array.isArray(tagMemories)) {
        return Promise.resolve(
          tagMemories.map((memory: any, index: number) => {
            // const title = memory.feature || `Item ${index + 1}`;
            const content = memory.value || JSON.stringify(memory, null, 2);
            const id = memory.id || `${element.memoryId}-${index}`;

            return new ProfileMemoryItem(
              content,
              "",
              vscode.TreeItemCollapsibleState.Collapsed,
              id,
              false,
              "profileMemory",
              memory
            );
          })
        );
      }
    }

    // If element is a memory item (not loading), return its details
    if (
      element.memoryId &&
      element.memoryId !== "loading" &&
      !String(element.memoryId).startsWith("tag-")
    ) {
      return Promise.resolve(this.createMemoryDetailItems(element));
    }

    return Promise.resolve([]);
  }

  private createMemoryDetailItems(
    memoryItem: ProfileMemoryItem
  ): ProfileMemoryItem[] {
    const details: ProfileMemoryItem[] = [];

    // Add metadata details if available
    if (memoryItem.memoryId) {
      details.push(
        new ProfileMemoryItem(
          "ID",
          memoryItem.memoryId,
          vscode.TreeItemCollapsibleState.None,
          `${memoryItem.memoryId}-id`,
          true
        )
      );
    }

    // Add content detail
    if (
      memoryItem.content &&
      memoryItem.content !== "empty" &&
      memoryItem.content !== "error"
    ) {
      details.push(
        new ProfileMemoryItem(
          "Content",
          memoryItem.content,
          vscode.TreeItemCollapsibleState.None,
          `${memoryItem.memoryId}-content`,
          true
        )
      );
    }

    if (memoryItem.source.feature) {
      details.push(
        new ProfileMemoryItem(
          "Feature",
          memoryItem.source.feature,
          vscode.TreeItemCollapsibleState.None,
          `${memoryItem.memoryId}-feature`,
          true
        )
      );
    }

    if (memoryItem.source.timestamp) {
      details.push(
        new ProfileMemoryItem(
          "Timestamp",
          memoryItem.source.timestamp,
          vscode.TreeItemCollapsibleState.None,
          `${memoryItem.memoryId}-timestamp`,
          true
        )
      );
    }

    return details;
  }

  private async loadMemories(): Promise<void> {
    this._isLoading = true;
    this._onDidChangeTreeData.fire();

    try {
      const response = await apiClient.getProfileMemory();
      if (response.data.success) {
        this._memories = this.parseMemories(response.data.data.profile_memory);
      } else {
        throw new Error(response.data.message);
      }
    } catch (error) {
      console.error("Failed to load profile memories:", error);
      this._memories = [
        new ProfileMemoryItem(
          "Error loading profile memories",
          "error",
          vscode.TreeItemCollapsibleState.None
        ),
      ];
    } finally {
      this._isLoading = false;
      this._onDidChangeTreeData.fire();
    }
  }

  private parseMemories(data: any): ProfileMemoryItem[] {
    console.log("Raw Profile Memory API data:", JSON.stringify(data, null, 2));

    if (!data) {
      return [
        new ProfileMemoryItem(
          "No profile memories found",
          "empty",
          vscode.TreeItemCollapsibleState.None
        ),
      ];
    }

    let memories: any[] = [];

    // Handle different data structures
    if (Array.isArray(data)) {
      memories = data;
    }

    console.log("Parsed profile memories:", memories);

    if (memories.length === 0) {
      return [
        new ProfileMemoryItem(
          "No profile memories found",
          "empty",
          vscode.TreeItemCollapsibleState.None
        ),
      ];
    }

    // Group memories by tag
    const tagGroups = new Map<string, any[]>();

    memories.forEach((memory) => {
      const tag = memory.tag || "Uncategorized";
      if (!tagGroups.has(tag)) {
        tagGroups.set(tag, []);
      }
      tagGroups.get(tag)!.push(memory);
    });

    // Create tag group items
    const tagGroupItems: ProfileMemoryItem[] = [];

    tagGroups.forEach((tagMemories, tag) => {
      // Create a tag group item
      const tagGroupItem = new ProfileMemoryItem(
        `${tag}`,
        `${tagMemories.length} item(s)`,
        vscode.TreeItemCollapsibleState.Collapsed,
        `tag-${tag}`,
        false,
        "tag-group"
      );

      // Store the memories for this tag in the tagGroupItem
      (tagGroupItem as any).tagMemories = tagMemories;

      tagGroupItems.push(tagGroupItem);
    });

    // Sort tag groups alphabetically
    tagGroupItems.sort((a, b) => a.label.localeCompare(b.label));

    return tagGroupItems;
  }
}

export class ProfileMemoryItem extends vscode.TreeItem {
  constructor(
    public readonly label: string,
    public readonly content: string,
    public readonly collapsibleState: vscode.TreeItemCollapsibleState,
    public readonly memoryId?: string,
    public readonly isDetail: boolean = false,
    public readonly contextValue: string = "profileMemory",
    public readonly source?: any
  ) {
    super(label, collapsibleState);

    // Set tooltip to show full content
    try {
      // Ensure content is a string and not null/undefined
      const safeContent =
        content && typeof content === "string"
          ? content
          : String(content || "");
      this.tooltip = new vscode.MarkdownString(safeContent);
    } catch (error) {
      // Fallback to plain text if MarkdownString fails
      this.tooltip =
        content && typeof content === "string"
          ? content
          : String(content || "");
    }

    // Set description to show a preview
    if (
      content &&
      content !== "empty" &&
      content !== "error" &&
      typeof content === "string"
    ) {
      this.description =
        content.length > 50 ? content.substring(0, 50) + "..." : content;
    } else {
      this.description =
        content && typeof content === "string"
          ? content
          : String(content || "");
    }

    this.source = source;
    this.contextValue = contextValue;

    // Add command based on item type
    if (this.memoryId === "loading") {
      this.iconPath = new vscode.ThemeIcon("loading");
    } else if (this.isDetail) {
      // Detail items show info icon, no command needed
    } else if (contextValue === "tag-group") {
      // Tag group items show tag icon
      this.iconPath = new vscode.ThemeIcon("tag");
    } else {
      // Main memory items are collapsible, no command needed
      this.iconPath = new vscode.ThemeIcon("person");
    }
  }
}

import * as vscode from "vscode";
import { apiClient } from "./apiClient";
import { getProjectName, getProjectNameWithPrefix } from "./utils";

export class EpisodicMemoryTreeProvider
  implements vscode.TreeDataProvider<EpisodicMemoryItem>
{
  private _onDidChangeTreeData: vscode.EventEmitter<
    EpisodicMemoryItem | undefined | null | void
  > = new vscode.EventEmitter<EpisodicMemoryItem | undefined | null | void>();
  readonly onDidChangeTreeData: vscode.Event<
    EpisodicMemoryItem | undefined | null | void
  > = this._onDidChangeTreeData.event;

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
      "Are you sure you want to clear all episodic memories? This action cannot be undone.",
      { modal: true },
      "Clear"
    );

    if (result !== "Clear") {
      return;
    }

    try {
      await apiClient.deleteMemory();
      vscode.window.showInformationMessage(
        "Episodic memories cleared successfully"
      );
      this.refresh();
    } catch (error) {
      console.error("Failed to clear episodic memories:", error);
      vscode.window.showErrorMessage("Failed to clear episodic memories");
    }
  }

  getTreeItem(element: EpisodicMemoryItem): vscode.TreeItem {
    return element;
  }

  getChildren(element?: EpisodicMemoryItem): Thenable<EpisodicMemoryItem[]> {
    if (!element) {
      const items: EpisodicMemoryItem[] = [];

      // Add loading indicator if refreshing
      if (this._isLoading) {
        items.push(
          new EpisodicMemoryItem(
            "Loading...",
            "Refreshing episodic memories...",
            vscode.TreeItemCollapsibleState.None,
            "loading"
          )
        );
      }

      // Add memories
      items.push(...this._memories);

      return Promise.resolve(items);
    }

    // If element is a memory item (not loading), return its details
    if (element.memoryId && element.memoryId !== "loading") {
      return Promise.resolve(this.createMemoryDetailItems(element));
    }

    return Promise.resolve([]);
  }

  private createMemoryDetailItems(
    memoryItem: EpisodicMemoryItem
  ): EpisodicMemoryItem[] {
    const details: EpisodicMemoryItem[] = [];

    // Add metadata details if available
    if (memoryItem.memoryId) {
      details.push(
        new EpisodicMemoryItem(
          "ID",
          memoryItem.memoryId,
          vscode.TreeItemCollapsibleState.None,
          `${memoryItem.memoryId}-id`,
          true
        )
      );
    }
    if (memoryItem.source) {
      if (memoryItem.source.content_type) {
        details.push(
          new EpisodicMemoryItem(
            "Content Type",
            memoryItem.source.content_type,
            vscode.TreeItemCollapsibleState.None,
            `${memoryItem.memoryId}-id`,
            true
          )
        );
      }
    }

    // Add content detail
    if (memoryItem.content) {
      details.push(
        new EpisodicMemoryItem(
          "Content",
          memoryItem.content,
          vscode.TreeItemCollapsibleState.None,
          `${memoryItem.memoryId}-content`,
          true
        )
      );
    }

    // Add timestamp if available (you can extend this based on your API response)
    if (memoryItem.source.timestamp) {
      details.push(
        new EpisodicMemoryItem(
          "Timestamp",
          memoryItem.source.timestamp,
          vscode.TreeItemCollapsibleState.None,
          `${memoryItem.memoryId}-timestamp`,
          true
        )
      );
    }

    if (memoryItem.source.group_id) {
      details.push(
        new EpisodicMemoryItem(
          "Group ID",
          memoryItem.source.group_id,
          vscode.TreeItemCollapsibleState.None,
          `${memoryItem.memoryId}-group-id`,
          true
        )
      );
    }

    if (memoryItem.source.producer_id) {
      details.push(
        new EpisodicMemoryItem(
          "Producer ID",
          memoryItem.source.producer_id,
          vscode.TreeItemCollapsibleState.None,
          `${memoryItem.memoryId}-producer-id`,
          true
        )
      );
    }

    if (memoryItem.source.produced_for_id) {
      details.push(
        new EpisodicMemoryItem(
          "Produced For ID",
          memoryItem.source.produced_for_id,
          vscode.TreeItemCollapsibleState.None,
          `${memoryItem.memoryId}-produced-for-id`,
          true
        )
      );
    }

    details.push(
      new EpisodicMemoryItem(
        "Source",
        JSON.stringify(memoryItem.source, null, 2),
        vscode.TreeItemCollapsibleState.None,
        `${memoryItem.memoryId}-source`,
        true
      )
    );

    return details;
  }

  private async loadMemories(): Promise<void> {
    this._isLoading = true;
    this._onDidChangeTreeData.fire();

    try {
      const response = await apiClient.getEpisodicMemory();
      if (response.data.success) {
        this._memories = this.parseMemories(response.data.data.episodic_memory);
      } else {
        throw new Error(response.data.message);
      }
    } catch (error) {
      console.error("Failed to load episodic memories:", error);
      this._memories = [
        new EpisodicMemoryItem(
          "Error loading memories",
          "error",
          vscode.TreeItemCollapsibleState.None
        ),
      ];
    } finally {
      this._isLoading = false;
      this._onDidChangeTreeData.fire();
    }
  }

  private parseMemories(data: any): EpisodicMemoryItem[] {
    console.log("data:", data);
    console.log("Raw API data:", JSON.stringify(data, null, 2));

    if (!data) {
      return [
        new EpisodicMemoryItem(
          "No memories found",
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

    console.log("Parsed memories:", memories);

    if (memories.length === 0) {
      return [
        new EpisodicMemoryItem(
          "No memories found",
          "empty",
          vscode.TreeItemCollapsibleState.None
        ),
      ];
    }

    return memories.map((memory, index) => {
      const title = memory.content;
      const content = memory.content || "-";
      const id = memory.uuid;

      console.log(`Memory ${index + 1}:`, {
        title,
        content: content.substring(0, 100) + "...",
        id,
      });

      return new EpisodicMemoryItem(
        content,
        "",
        vscode.TreeItemCollapsibleState.Collapsed,
        id,
        false,
        memory
      );
    });
  }
}

export class EpisodicMemoryItem extends vscode.TreeItem {
  constructor(
    public readonly label: string,
    public readonly content: string,
    public readonly collapsibleState: vscode.TreeItemCollapsibleState,
    public readonly memoryId?: string,
    public readonly isDetail: boolean = false,
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
    // Add command based on item type
    if (this.memoryId === "loading") {
      this.iconPath = new vscode.ThemeIcon("loading");
    } else if (this.isDetail) {
      // this.iconPath = new vscode.ThemeIcon('info');
    } else {
      this.iconPath = new vscode.ThemeIcon("book");
    }
  }
}

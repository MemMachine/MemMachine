# MemMachine

This is the extension for MemMachine, which lets you leverage its capabilities within the Cursor editor.

## Features

MemMachine provides powerful memory management capabilities directly within your Cursor editor environment.

### Memory Panels

The extension includes two dedicated panels for viewing different types of memories retrieved from the MemMachine API:

#### Episodic Memory Panel
- A list of episodic memories with titles, content previews, and metadata
#### Profile Memory Panel
- A list of profile memories with titles, content previews, and metadata

### Available Commands

- **MemMachine: Show Episodic Memory** - Opens the episodic memory panel
- **MemMachine: Show Profile Memory** - Opens the profile memory panel
- **MemMachine: Refresh All Memories** - Refreshes both episodic and profile memory panels
- **MemMachine: Register MCP Server** - Register a new MCP server for MemMachine integration
- **MemMachine: Unregister MCP Server** - Remove a registered MCP server
- **MemMachine: Auto-register Default Server** - Automatically register the default MemMachine MCP server

## Requirements

- Cursor editor or VS Code
- VS Code engine version 1.93.0 or higher
- Cursor version 0.40.0 or higher (for Cursor-specific MCP features)

## Installation

1. Install the extension from the Cursor marketplace or load it locally


## Usage

### Accessing the Memory Panels

1. **Via Command Palette:**
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
   - Type "MemMachine: Show Episodic Memory" or "MemMachine: Show Profile Memory"
   - Select the command to focus the desired panel

2. **Via Bottom Panel:**
   - Look for the "MemMachine" tab in the bottom panel area
   - Click on "Episodic Memory" to view your episodic memories
   - Click on "Profile Memory" to view your profile memories

Both memory panels appear in the bottom panel area, alongside other panels like Output, Problems, Terminal, etc.

### MCP Server Configuration

The extension supports Model Context Protocol (MCP) server registration for both Cursor and VS Code:

#### Automatic Registration
- The extension automatically registers the default MemMachine MCP server on activation
- Default server URL: `http://18.116.238.202:8001/mcp/`
- Default authentication: Bearer token (configurable in config file)

#### Manual Registration
1. Open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`)
2. Run "MemMachine: Register MCP Server"

#### Environment Detection
- The extension automatically detects whether it's running in Cursor or VS Code
- Uses appropriate APIs for each environment:
  - **Cursor**: Uses `vscode.cursor.mcp` API for native MCP integration
  - **VS Code**: Uses `vscode.mcp` API or falls back to settings configuration

#### Unregistering Servers
1. Open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`)
2. Run "MemMachine: Unregister MCP Server"
3. Select the server to remove from the list

#### Configuration
All MCP server settings are centralized in the `src/config.ts` file:
- **API Base URL**: `http://18.116.238.202:8001/`
- **Server URL**: `http://18.116.238.202:8001/mcp/`
- **Authentication Token**: `your-auth-token-here`

## Release Notes

### 0.0.1

Initial release of MemMachine extension for Cursor editor.

## Contributing

This extension is developed by MemVerge. For issues or feature requests, please contact the development team.

## License

This extension is proprietary software developed by MemVerge.

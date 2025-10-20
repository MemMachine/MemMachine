import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Download, ExternalLink, Settings, Info, Zap } from 'lucide-react'

export default function Help() {
  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
    // You could add a toast notification here
  }

  return (
    <div className="container mx-auto p-6 max-w-4xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight">MemMachine Extension Help</h1>
        <p className="text-muted-foreground mt-2">
          Get started with MemMachine Extension for Cursor
        </p>
      </div>

      <div className="grid gap-6">
        {/* Project Introduction */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Info className="h-5 w-5" />
              About MemMachine Extension
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-muted-foreground">
              MemMachine Extension is a powerful memory management system that provides intelligent memory capabilities 
              directly within your Cursor editor. It helps you maintain context, store important information, 
              and enhance your coding experience with persistent memory across sessions.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-semibold mb-2">Episodic Memory</h4>
                <p className="text-sm text-muted-foreground">
                  Stores conversation history, project context, and session-specific information 
                  to provide continuity across your coding sessions.
                </p>
              </div>
              <div>
                <h4 className="font-semibold mb-2">Profile Memory</h4>
                <p className="text-sm text-muted-foreground">
                  Maintains your preferences, coding patterns, and personalized settings 
                  to deliver tailored assistance.
                </p>
              </div>
            </div>

            <div className="bg-muted p-4 rounded-md">
              <h4 className="font-semibold mb-2 flex items-center gap-2">
                <Zap className="h-4 w-4" />
                Key Features
              </h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Intelligent memory management via Model Context Protocol (MCP)</li>
                <li>• Seamless integration with Cursor editor</li>
                <li>• Persistent storage across sessions</li>
              </ul>
            </div>

            <div className="rounded-md">
            <div className="grid gap-6">
              <div>
                <p className="text-sm text-muted-foreground mb-4">
                  The MemMachine Extension provides various memory management features accessible through the sidebar and command palette.
                </p>
                <div className="border rounded-lg overflow-hidden">
                  <img 
                    src="/extention_features.png" 
                    alt="MemMachine Extension Features" 
                    className="w-full h-auto"
                  />
                </div>
              </div>

              <div>
                <h4 className="font-semibold mb-3">Extension Configuration</h4>
                <p className="text-sm text-muted-foreground mb-4">
                  Configure the extension settings including API base URL and authentication token in the Cursor settings panel.
                </p>
                <div className="border rounded-lg overflow-hidden">
                  <img 
                    src="/extention_configuration.png" 
                    alt="MemMachine Extension Configuration" 
                    className="w-full h-auto"
                  />
                </div>
              </div>
            </div>
            </div>
          </CardContent>
        </Card>

        {/* Extension Download */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Download className="h-5 w-5" />
              Download & Install Extension
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-muted-foreground">
              Download and install the MemMachine Extension to start using memory management features in Cursor.
            </p>

            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 border rounded-md">
                <div>
                  <h4 className="font-semibold">MemMachine Extension</h4>
                  <p className="text-sm text-muted-foreground">Version 0.0.4 • Published by MemVerge</p>
                </div>
                <Button asChild>
                  <a 
                    href="/memmachine-0.0.4.vsix" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="flex items-center gap-2"
                  >
                    <Download className="h-4 w-4" />
                    Download Extension
                    <ExternalLink className="h-4 w-4" />
                  </a>
                </Button>
              </div>

              <div className="bg-muted p-4 rounded-md">
                <h4 className="font-semibold mb-2">System Requirements</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Cursor editor or VS Code</li>
                  <li>• VS Code engine version 1.93.0 or higher</li>
                  <li>• Cursor version 0.40.0 or higher (for MCP features)</li>
                </ul>
              </div>

              <div className="bg-muted p-4 rounded-md">

            <div className="space-y-4">
              <div>
                <h4 className="font-semibold mb-3">How to Install the Extension</h4>
                <ol className="text-sm text-muted-foreground space-y-2 mb-4">
                  <li>1. Download the extension file (memmachine-0.0.4.vsix) from the download section above</li>
                  <li>2. Open the Command Palette in Cursor (Cmd+Shift+P on macOS or Ctrl+Shift+P on Windows/Linux)</li>
                  <li>3. Type "Extensions: Install from VSIX..." and select the command</li>
                  <li>4. Navigate to and select the downloaded memmachine-0.0.4.vsix file</li>
                  <li>5. The extension will be installed and appear in your extensions list</li>
                </ol>
                
                <div className="border rounded-lg overflow-hidden">
                  <img 
                    src="/extention_install.png" 
                    alt="How to Install MemMachine Extension from VSIX" 
                    className="w-full h-auto"
                  />
                </div>
              </div>

              <div className="bg-blue-50 dark:bg-blue-950/30 p-4 rounded-md border border-blue-200 dark:border-blue-800">
                <p className="text-sm text-blue-800 dark:text-blue-200">
                  <strong>Tip:</strong> After installation, you'll need to configure the MCP settings as shown in the next section to enable full functionality.
                </p>
              </div>
            </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* MCP Setup Guide */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              How to Setup MCP in Cursor
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-muted-foreground">
              Model Context Protocol (MCP) enables MemMachine Extension to communicate with Cursor. 
              Follow these steps to configure MCP integration.
            </p>

            <div className="space-y-6">
              <div>
                <h4 className="font-semibold mb-3">Step 1: Configure MCP Settings</h4>
                <p className="text-sm text-muted-foreground mb-3">
                  Open your Cursor settings and add the following MCP configuration:
                </p>
                <div className="bg-muted p-4 rounded-md">
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-sm font-medium">mcp.json Configuration</span>
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => copyToClipboard(`{
  "mcpServers": {
    "memmachine": {
      "url": "http://ec2-18-223-182-61.us-east-2.compute.amazonaws.com:8001/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_AUTH_TOKEN_HERE"
      }
    }
  }
}`)}
                    >
                      Copy
                    </Button>
                  </div>
                  <pre className="text-xs text-muted-foreground overflow-x-auto">
{`{
  "mcpServers": {
    "memmachine": {
      "url": "http://ec2-18-223-182-61.us-east-2.compute.amazonaws.com:8001/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_AUTH_TOKEN_HERE"
      }
    }
  }
}`}
                  </pre>
                </div>
              </div>

              <div>
                <h4 className="font-semibold mb-3">Step 2: Setup Authentication</h4>
                <p className="text-sm text-muted-foreground mb-3">
                  Replace "your-auth-token-here" with your actual authentication token from the MemMachine Extension dashboard.
                </p>
                <div className="bg-blue-50 dark:bg-blue-950/30 p-4 rounded-md border border-blue-200 dark:border-blue-800">
                  <p className="text-sm text-blue-800 dark:text-blue-200">
                    <strong>Note:</strong> You can find your authentication token on your user profile page in this MemMachine Extension dashboard.
                  </p>
                </div>
              </div>

              <div>
                <h4 className="font-semibold mb-3">Step 3: Restart Cursor</h4>
                <p className="text-sm text-muted-foreground">
                  After configuring MCP settings, restart Cursor to load the MemMachine Extension integration.
                  The extension will automatically register the MCP server and you'll be ready to use MemMachine Extension features.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

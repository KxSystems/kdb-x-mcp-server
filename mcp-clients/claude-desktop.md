# Claude Desktop Configuration Guide

This guide explains how to configure Claude Desktop with the KDB-X MCP Server.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Validation](#validation)
- [Developer Mode](#developer-mode)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Additional Resources](#additional-resources)

## Overview

Claude Desktop is MCP Client from Anthropic that supports Model Context Protocol (MCP) servers. This configuration allows you to use the KDB-X MCP Server directly with Claude Desktop, enabling natural language interactions with your KDB-X database.

## Prerequisites

- **Claude Desktop installed** (macOS or Windows) - [Download here](https://claude.ai/download)
- **NPX installed** (required for streamable-http transport) - comes bundled with [Node.js](https://nodejs.org/en)
- **KDB-X MCP Server installed** - See the [main README](../README.md#mcp-server-installation) for installation instructions

## Configuration

Claude Desktop requires a `claude_desktop_config.json` file to be configured.

### Configuration File Location

| Platform | Default Configuration File Location                               |
| -------- | ----------------------------------------------------------------- |
| macOS    | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows  | `%APPDATA%\Claude\claude_desktop_config.json`                     |

### Option 1: streamable-http Transport

This option requires manually starting the MCP server in a separate terminal.

Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "kdbx-streamable": {
      "command": "npx",
      "args": [
         "mcp-remote",
         "http://localhost:8000/mcp"
      ]
    }
  }
}
```

**Note:**
- You must have `npx` installed and available on your PATH
- You must start the MCP Server manually (in a separate terminal) before using it: `uv run mcp-server`
- The server must remain running while you use it
- Default port is 8000 (can be changed with `--mcp.port`)
- MCP logs will be visible from your terminal
- Ensure you have the correct endpoint - update the port if you configured a different one

### Option 2: stdio Transport

This option allows Claude Desktop to automatically start and stop the MCP server.

Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "kdbx-stdio": {
      "command": "/Users/<user>/.local/bin/uv",
      "args": [
        "--directory",
        "/path/to/this/repo/",
        "run",
        "mcp-server",
        "--mcp.transport",
        "stdio"
      ]
    }
  }
}
```

**Note:**
- Update `<user>` to point to the absolute path of the uv executable - only required if `uv` is not on your PATH
- Update the `--directory` path to the absolute path of this repo
- Currently `KDB-X` does not support Windows, meaning `stdio` is not an option for Windows users
- Claude Desktop is responsible for starting/stopping the MCP server when using `stdio`
- When using `stdio` the MCP logs will be available at [Claude Desktop's MCP Log Location](#claude-log-locations)

### Multiple MCP Servers

You can include multiple MCP servers in your configuration:

```json
{
  "mcpServers": {
    "kdbx-streamable": {
      "command": "npx",
      "args": [
         "mcp-remote",
         "http://localhost:8000/mcp"
      ]
    },
    "another-server": {...}
  }
}
```

For detailed setup instructions, see the [official Claude Desktop documentation](https://claude.ai/docs/desktop).

### Advanced Configuration Options

You can pass additional environment variables or command-line arguments to customize the MCP server behavior:

```json
{
  "mcpServers": {
    "kdbx-stdio": {
      "command": "/Users/<user>/.local/bin/uv",
      "args": [
        "--directory",
        "/absolute/path/to/kdb-x-mcp-server",
        "run",
        "mcp-server",
        "--mcp.transport",
        "stdio",
        "--db.host",
        "localhost",
        "--db.port",
        "8001"
      ],
      "env": {
        "KDBX_MCP_LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

For all available configuration options, see the [Command Line Tool](../README.md#command-line-tool) section in the main README.

## Validation

Follow these steps to verify your configuration is working correctly:

1. **Start the server** (if using `streamable-http` transport):

   ```bash
   uv run mcp-server
   ```

   If using `stdio`, skip this step - Claude Desktop will auto-start the server.

2. **Verify configuration file**:
   - Open `claude_desktop_config.json` in your editor
   - Ensure there are no syntax errors in the JSON
   - Windows users: make sure to quit Claude Desktop via the system tray before restarting

3. **Restart Claude Desktop**:
   - Completely quit and restart Claude Desktop
   - Navigate to `File` > `Settings` > `Developer`
   - You should see that your KDB-X MCP Server is running

   ![MCP Server Status](../screenshots/claude_mcp.png)

4. **Verify tools are available**:
   - From a chat window, click the `search and tools` icon just below the message box on the left
   - You'll see your MCP server listed (e.g., `kdbx-streamable`)
   - Click it to access all available tools

   ![Tools Available](../screenshots/claude_tools.png)

5. **Verify prompts and resources are available**:
   - Click the '+' in the chat window
   - Select `Add from kdbx-streamable` (or your configured server name)
   - You should see the list of available prompts and resources

   ![Prompts and Resources](../screenshots/claude_prompts_resources.png)

6. **Check server status**:
   - The KDB-X MCP server should appear in `File` > `Settings` > `Developer`
   - Server status should show as connected/running

## Developer Mode

Developer mode provides quick access to useful debugging features:

- **MCP Server Reloads** - no need to quit Claude Desktop for every MCP Server restart
- **MCP Configuration** - shortcut to your `claude_desktop_config.json`
- **MCP Logs** - shortcut to Claude Desktop MCP logs
  - Note: When using `streamable-http` transport, you'll also need to review the MCP logs from your terminal

### Enable Developer Mode

1. Start Claude Desktop
2. Click the menu in the upper-left corner > `Help` > `Troubleshooting` > `Enable Developer Mode`
3. Confirm any popups
4. Restart Claude Desktop
5. Click the menu in the upper-left corner > `Developer` - Developer settings should now be populated

## Usage

Once configured, you can interact with your KDB-X database using natural language in Claude Desktop. See the [Quickstart](../README.md#quickstart) guide for usage examples.

### Accessing MCP Tools

MCP tools are the primary way to interact with your KDB-X database. Tools are automatically available in Claude Desktop chat once the MCP server is connected.

**Method 1: Natural Language (Recommended)**

Simply describe what you want to do in natural language, and Claude will automatically select and invoke the appropriate KDB-X tools.

**Method 2: Manual Tool Selection**

If you want to explicitly select which tools Claude can use:

1. In the chat window, click the **search and tools** icon (hammer icon) below the message box on the left
2. You'll see your MCP server listed (e.g., `kdbx-streamable` or `kdbx-stdio`)
3. Click it to view and select specific tools
4. Selected tools will be highlighted and available for Claude to use in the conversation

**Important Notes:**
- Tools are automatically detected when the MCP server is connected
- Claude will intelligently choose which tools to use based on your request
- You can verify tool availability by clicking the search and tools icon

### Accessing MCP Resources

The KDB-X MCP server provides resources that can be added as context to your chat conversations. Resources provide static information that can help Claude better understand your KDB-X setup.

**How to Access Resources:**

1. In the chat window, click the **+** button
2. Select **Add from [server-name]** (e.g., "Add from kdbx-streamable")
3. You'll see a list of available resources
4. Click on a resource to add it to your conversation context

### Accessing MCP Prompts

The KDB-X MCP server provides pre-defined prompts for common tasks. Prompts are reusable workflows that guides Claude through specific operations.

**How to Access Prompts:**

1. In the chat window, click the **+** button
2. Select **Add from [server-name]** (e.g., "Add from kdbx-streamable")
3. Browse the list of available prompts
4. Click on a prompt to use it

**If the prompt requires arguments:**
- Claude Desktop will display input fields for required parameters
- Fill in the values (e.g., `table_name`, `analysis_type`)
- Submit to execute the prompt with your provided arguments

## Troubleshooting

### Server not appearing in Claude Desktop

- Validate `claude_desktop_config.json` syntax
- Check Claude Desktop's MCP logs for error messages (see [Claude Log Locations](#claude-log-locations))
- Fully quit Claude Desktop (Windows: quit from system tray) and restart

### Connection issues with streamable-http transport

- Confirm the MCP server is running in a separate terminal and review the logs for any error messages
- Verify the port in `claude_desktop_config.json` matches the server port
- Verify `uv` is installed and use the absolute path to the executable

### stdio transport fails to start

- Run `uv sync` first to install dependencies (recommended for first-time setup)
- Check the absolute path to the repository is correct
- Verify `uv` is installed and use the absolute path to the executable
- Check Claude Desktop's MCP logs for error messages (see [Claude Log Locations](#claude-log-locations))

### MCP Server disabled in Claude Desktop

- If you see that the MCP server is disabled after a query, restart/exit Claude Desktop and try again.

### Tools not showing or not being triggered

- Verify the server is listed and connected in `File` > `Settings` > `Developer`
- Restart Claude Desktop if the configuration was just added
- Check that tools are visible by clicking the `search and tools` icon

### Claude Log Locations

| Platform    | Path                             | Monitor Command                                               |
| ----------- | -------------------------------- | ------------------------------------------------------------- |
| **macOS**   | `~/Library/Logs/Claude/mcp*.log` | `tail -f ~/Library/Logs/Claude/mcp*.log`                      |
| **Windows** | `%APPDATA%\Claude\Logs\mcp*.log` | `Get-Content -Path "$env:APPDATA\Claude\Logs\mcp*.log" -Wait` |

### Claude Limits

You may need to upgrade to a paid plan to avoid Claude usage errors like:

> Claude hit the maximum length for this conversation. Please start a new conversation to continue chatting with Claude.

## Additional Resources

- [Official Claude Desktop Documentation](https://claude.ai/docs/desktop)
- [Official Claude MCP Documentation](https://modelcontextprotocol.io/quickstart/user)
- [KDB-X MCP Server Main README](../README.md)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [KDB-X documentation](https://docs.kx.com/public-preview/kdb-x/home.htm)

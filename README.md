# KDB-X MCP Server

This server enables end users to query KDB-X data through natural language, providing production-grade resources, prompts, and tools for seamless data interaction.

Built on an extensible framework with configurable templates, it allows for intuitive extension with custom integrations tailored to your specific needs.

The server leverages a combination of curated resources, intelligent prompts, and robust tools to provide appropriate guardrails and guidance for both users and AI models interacting with KDB-X.

## Table of Contents

- [Supported Environments](#supported-environments)
- [Prerequisites](#prerequisites)
- [Quickstart](#quickstart)
- [Features](#features)
- [KDB-X Setup](#kdb-x-setup)
- [MCP Server Installation](#mcp-server-installation)
- [Transport Options](#transport-options)
- [Command line Parameters](#command-line-parameters)
- [Usage with Claude Desktop](#usage-with-claude-desktop)
- [Prompts/Resources/Tools](#promptsresourcestools)
- [Development](#development)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Useful Resources](#useful-resources)

## Supported Environments

The following table shows the install options for supported Operating Systems:

| **Primary OS** | **KDB-X** | **KDB+** | **MCP Server** | **UV/NPX** | **Claude Desktop** | **Alternative MCP Client** |
|--------|-------------|-------------|------------------------|------------|-------------------|-----------|
| **Mac** | ✅ Local | ✅ Local | ✅ Local | ✅ Local | ✅ Local (streamable-http/stdio) | ✅ [Other clients](https://modelcontextprotocol.io/clients) |
| **Linux** | ✅ Local | ✅ Local | ✅ Local | ✅ Local | ❌ Not supported | ✅ [Other clients](https://modelcontextprotocol.io/clients) |
| **WSL** | ✅ Local | ✅ Local | ✅ Local | ✅ Local | ❌ Not supported | ✅ [Other clients](https://modelcontextprotocol.io/clients) |
| **Windows** | ⚠️ WSL | ✅ Local | ⚠️ WSL | ✅ Local | ✅ Local (streamable-http only) | ✅ [Other clients](https://modelcontextprotocol.io/clients) |
| **Windows** | ⚠️ WSL | ✅ Local | ✅ Local | ✅ Local | ✅ Local (streamable-http only) | ✅ [Other clients](https://modelcontextprotocol.io/clients) |
| **Windows** | ⚠️ Remote Linux | ✅ Local | ✅ Local | ✅ Local | ✅ Local (streamable-http only) | ✅ [Other clients](https://modelcontextprotocol.io/clients) |

> The KDB-X MCP server can connect to one KDB Service -  either KDB-X or KDB+, not both. \
> The chosen KDB Service needs to be listening on a host and port that is accessible to the KDB-X MCP server.

- **KDB-X**: Mac/Linux/WSL only (no native Windows support)
- **KDB+**: Windows/Mac/Linux/WSL
- **MCP Server**: UV required (Windows/Mac/Linux/WSL)
- **Claude Desktop**: Windows/Mac only
- **UV**: Required for running the MCP Server
- **NPX**: Required for streamable-http transport with Claude Desktop
- **Stdio transport**: Only works when Claude Desktop and MCP Server are on same OS

## Prerequisites

Before installing and running the KDB-X MCP Server, ensure you have met the following requirements:

- [Cloned this repo](#clone-the-repository)
- A `KDB-X/KDB+` Service listening on a host and port that will be accessible to the MCP Server
  - See examples - [KDB-X Setup](#kdb-x-setup) / [KDB+ Setup](#kdb-setup)
  - KDB-X can be installed by signing up to the [kdb-x public preview](https://kdb-x.kx.com/sign-in) - see [KDB-X documentation](https://docs.kx.com/public-preview/kdb-x/home.htm) for supporting information
  - Windows users can run the KDB-X MCP Server on Windows and connect to a local KDB-X database via WSL or remote KDB-X database running on Linux
  - Windows users can run a local KDB-X database by installing KDB-X on [WSL](https://learn.microsoft.com/en-us/windows/wsl/install), and use the default [streamable-http transport](#transport-options) when running the [KDB-X MCP Server](#run-the-server) - both share the same localhost network.
  - For details on KDB-X usage restrictions see [documentation](https://docs.kx.com/product/licensing/usage-restrictions.htm#kdb-x-personal-trial-download)
- [UV Installed](https://docs.astral.sh/uv/getting-started/installation/) for running the KDB-X MCP Server - available on Windows/Mac/Linux/WSL
- [Claude Desktop](https://claude.ai/download) or another MCP-compatible client installed, that will connect the the KDB-X MCP Server - available on Windows/Mac
- [NPX](https://nodejs.org/en) is required to use `streamable-http` transport with Claude Desktop
  - `npx` may not be required if you are using a different MCP Client - consult the documentation of your chosen MCP Client
  - `npx` comes bundled with the [nodejs](https://nodejs.org/en) installer - available on Windows/Mac/Linux/WSL
  - See [example configuration with streamable-http](#example-configuration-with-streamable-http)

## Quickstart

To demonstrate basic usage of the KDB-X MCP Server, using an empty KDB-X database, follow the quickstart steps below.

> Note: Ensure you have followed the necessary [prerequisites steps](#prerequisites)

1. Open a KDB-X service listening on a port.

   By default the KDB-X MCP server will connect to KDB-X service on port 5000 - [but this can be changed](#command-line-parameters) via command line flags or environment variables.

   > Note: KDB-X is currently not supported on Windows - if you are using Windows we recommend running KDB-X on WSL as outlined in the [prerequisites steps](#prerequisites)

   ```bash
   q -p 5000
   ```

2. Load the sql interface.

   ```q
   .s.init[]
   ```

3. Add a dummy table e.g. `trade`.

   ```q
   rows:10000;
   trade:([]time:.z.d+asc rows?.z.t;sym:rows?`AAPL`GOOG`MSFT`TSLA`AMZN;price:rows?100f;size:rows?1000);
   ```

4. [Configure Claude Desktop](#configure-claude-desktop) with your chosen transport.

5. [Start your MCP server](#mcp-server-installation).

   If you have configured Claude Desktop with [stdio transport](#example-configuration-with-stdio), then this step is not required. Please move directly to step 6 (Claude Desktop will manage starting the MCP Server for you).

   ```bash
   uv run mcp-server
   ```

6. Start Claude Desktop and verify that the tools and prompts outlined in the [Validate Claude Desktop Config](#validate-claude-desktop-config) section are visible.

7. **Load database context**: Select the `kdbx_describe_tables` and `kdbx_sql_query_guidance` [resources](#resources) to add them to your conversation. This will give your MCP client an overview of your database structure and available tables, along with guidance on writing effective SQL queries.

8. **Explore specific tables**: Use the `kdbx_table_analysis` [prompt](#prompts) to get detailed analysis and insights about individual tables in your database.

9. **Ask questions in natural language**: Interact with your KDB-X database using plain English. Your MCP client will automatically use the `kdbx_run_sql_query` [tool](#tools) to execute the appropriate queries based on your requests.

## Features

- **SQL Interface to KDB-X**: Run SELECT SQL queries against KDB-X databases
- **Built-In Query Safety Protection**: Automatic detection and blocking of dangerous SQL operations like INSERT,DROP,DELETE etc.
- **Smart Query Result Optimization**: Smart result truncation (max 1000 rows) with clear messaging about data limits
- **SQL Query Guidance for LLM**: Comprehensive LLM-ready MCP resource (file://guidance/kdb-sql-queries) with syntax examples and best practices
- **Database Schema Discovery**: Explore and understand your database tables and structure using the included MCP resource for quick, intelligent insights.
- **Auto-Discovery System**: Automatic discovery and registration of tools, resources, and prompts from their respective directories
- **Resilient Connection Management**: Robust KDB-X connection handling with automatic retry logic and connection caching
- **Ready-Made Extension Template**: Ready-to-use templates for tools, resources, and prompts with best practices and documentation for extending functionality
- **Unified Intelligence: Prompts, Tools & MCP Resources Working Together**: A powerful combination of intelligent prompts, purpose-built tools, and curated MCP resources—all working together to deliver fast, optimized, and context-aware results.
- **HTTP Streamable Protocol Support**: Supports the latest MCP streamable HTTP protocol for efficient data flow, while automatically blocking the deprecated SSE protocol.

## KDB-X Setup

The KDB-X MCP server connects to the KDB-X service on a designated host and port.

To start the KDB-X service and make it accessible locally you can run:

```bash
q -p 5000
```

The KDB-X MCP server communicates with the KDB-X service using its SQL interface.

Load the SQL interface:

```q
.s.init[]
```

## KDB+ Setup

The KDB-X MCP server connects to the KDB+ service on a designated host and port.

To start the KDB+ service and make it accessible locally you can run:

```bash
q -p 5000
```

The KDB-X MCP server communicates with the KDB+ service using its SQL interface.

When using KDB+, an [s.k_](https://code.kx.com/insights/1.14/core/sql.html#sql-language-support) file must be present in your `QHOME` - This file comes bundled with insights core

Load the SQL interface:

```q
\l s.k_
```

## MCP Server Installation

The MCP Server can be installed on Windows, Mac, Linux and [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)

### Clone the repository

```bash
git clone https://github.com/KxSystems/kdb-x-mcp-server.git
cd kdb-x-mcp-server
```

### Run the server

The server will start with `streamable-http` transport by default.

For Windows users with WSL installed -  when using `streamable-http` transport, the MCP Server can run on either Windows or WSL. For both scenarios, the MCP Server will be available on the same shared localhost network. Claude Desktop (running on Windows) will connect over `localhost`. So this repository can be cloned to either Windows or WSL. `uv` needs be installed on the same OS where the MCP Server will be running.

If you are using `stdio` on Windows, Claude Desktop will manage starting and stopping the MCP server. So this repository will need to be cloned to the Windows filesystem. `uv` will need be installed on Windows.

```bash
uv run mcp-server
```

## Transport Options

For more info on the supported transports see official documentation

- [streamable-http](https://modelcontextprotocol.io/docs/concepts/transports#streamable-http)
- [stdio](https://modelcontextprotocol.io/docs/concepts/transports#standard-input%2Foutput-stdio)

> Note: We don't support [sse](https://modelcontextprotocol.io/docs/concepts/transports#server-sent-events-sse-deprecated) transport (server-sent events) as it has been deprecated since protocol version 2024-11-05.

## Command line Parameters

```bash
uv run mcp-server -h
usage: mcp-server [-h] [--streamable-http | --stdio] [--kdbx-mcp-port KDBX_MCP_PORT]
                  [--kdbx-host KDBX_HOST] [--kdbx-port KDBX_PORT]
                  [--kdbx-timeout KDBX_TIMEOUT] [--kdbx-retry KDBX_RETRY]

KDB-X MCP Server

options:
  -h, --help            show this help message and exit
  --streamable-http     Start the KDB-X MCP server with streamable HTTP transport (default)
  --stdio               Start the KDB-X MCP server with stdio transport
  --kdbx-mcp-port KDBX_MCP_PORT
                        Port number the KDB-X MCP server will listen on when using streamable-http
                        transport (default 8000)
  --kdbx-host KDBX_HOST
                        KDB-X host that the MCP server will connect to (default: localhost)
  --kdbx-port KDBX_PORT
                        KDB-X port that the MCP server will connect to (default: 5000)
  --kdbx-timeout KDBX_TIMEOUT
                        KDB-X connection timeout in seconds (default: 1)
  --kdbx-retry KDBX_RETRY
                        KDB-X connection retry attempts (default: 2)
```

> Note: `kdbx-*` command line flags can be used when pointing to a KDB+ Service

**Environment Variables:**

- `KDBX_MCP_TRANSPORT`: Set transport mode (streamable-http, stdio)
- `KDBX_MCP_PORT`: Set port number (default: 8000)
- `KDBX_MCP_HOST`: Set host address (default: 127.0.0.1)
- `KDBX_MCP_SERVER_NAME`: Set server name (default: KDB-X_Demo)
- `KDBX_LOG_LEVEL`: Set logging level (default: INFO)
- `KDBX_RETRY`: KDB-X server connection retry count (default: 2)
- `KDBX_TIMEOUT`: KDB-X server connection timeout in seconds (default: 2)
- `KDBX_HOST`: KDB-X server hostname (default: localhost)
- `KDBX_PORT`: KDB-X server port (default: 5000)
- `KDBX_USERNAME`: KDB-X username (optional)
- `KDBX_PASSWORD`: KDB-X password (optional)

> Note: `KDBX_*` environment variables can be used when pointing to a KDB+ Service

**Configuration Priority:**

1. **CLI flags** (highest precedence) - `--streamable-http`, `--kdbx-port 8000`, `--kdbx-host myhost`
2. **Environment variables** (middle precedence) - `KDBX_MCP_TRANSPORT=streamable-http`, `KDBX_HOST=myhost`
3. **Default values** (lowest precedence)

## Usage with Claude Desktop

### Configure Claude Desktop

Claude Desktop requires a `claude_desktop_config.json` file to be available.

Add one of the example configs below, to the default configuration file location for your OS.

| Platform | Default Configuration File Location |
|----------|---------------------------|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |

### Example configuration with streamable-http

To configure Claude Desktop with KDB-X MCP Server using `streamable-http`, copy the below example into an empty `claude_desktop_config.json` file.

If you have pre-existing MCP servers see [example config with multiple mcp-servers](#example-configuration-with-multiple-mcp-servers)

```json
{
  "mcpServers": {
    "KDB-X MCP streamable": {
      "command": "npx",
      "args": [
         "mcp-remote",
         "http://localhost:8000/mcp"
      ]
    }
  }
}
```

**Note**

- To use `streamable-http` with Claude Desktop you must have `npx` installed and available on your path - you can install it via [nodejs.org](https://nodejs.org/en)
- You will need to start the MCP Server as a standalone python process. See section [Run the server](#run-the-server)
- Ensure you have the correct endpoint - in this example our KDB-X MCP server is running on port `8000`.
- This means you will be responsible for starting and stopping the MCP Server, Claude Desktop will only access it via `npx`
- MCP logs will be visible from your terminal

#### Example configuration with stdio

To configure Claude Desktop with KDB-X MCP Server using `stdio`, copy this into an empty `claude_desktop_config.json` file.

If you have pre-existing MCP servers see [example config with multiple mcp-servers](#example-configuration-with-multiple-mcp-servers)

```json
{
  "mcpServers": {
    "KDB-X MCP stdio": {
      "command": "/Users/<user>/.local/bin/uv",
      "args": [
        "--directory",
        "/path/to/this/repo/",
        "run",
        "mcp_server"
      ]
    }
  }
}
```

**Note**

- Update your `<user>` to point to the absolute path of the uv executable - only required if `uv` is on your path
- Update the `--directory` path to the absolute path of this repo
- Currently `KDB-X` does not support Windows, meaning `stdio` is not an option for Windows users
- Claude Desktop is responsible for starting/stopping the MCP server when using `stdio`
- When using `stdio` the MCP logs will be available at [Claude Desktop's MCP Log Location](#claude-log-locations)

#### Example configuration with multiple MCP servers

You can include multiple MCP servers like this

```json
{
  "mcpServers": {
    "KDB-X MCP streamable": {
      "command": "npx",
      "args": [
         "mcp-remote",
         "http://localhost:8000/mcp"
      ]
    },
    "Another MCP Server": {...}
  }
}
```

For detailed setup instructions, see the [official Claude Desktop documentation](https://claude.ai/docs/desktop).

### Validate Claude Desktop Config

1. If you are using `streamable-http` you will need to start the MCP Server in a separate terminal window, and ensure it remains running. If you are using `stdio` skip to step 2.
2. Once the `claude_desktop_config.json` has been added, with your chosen transport config, restart Claude Desktop. Then navigate to `File` > `Settings` > `Developer`. You should see that your KDB-X MCP Server is running.
   - Windows users: make sure to quit Claude Desktop via the system tray before restarting.

   ![alt text](screenshots/claude_mcp.png)
3. From a chat window click the `search and tools` icon just below the message box on the left. You’ll see your MCP server listed as `KDB-X MCP`. Click it to access the `kdbx_run_sql_query` tool.

   ![alt text](screenshots/claude_tools.png)
4. Click the '+' in the chat window, then select `Add from KDB-X MCP` to view the list of available resources.

   ![alt text](screenshots/claude_resources.png)


### Enable Claude Desktop Developer Mode

Developer mode can be enabled to give quick access to:

- MCP Server Reloads - No need to quit Claude Desktop for every MCP Server restart
- MCP Configuration - Shortcut to your `claude_desktop_config.json`
- MCP Logs - Shortcut to Claude Desktop MCP logs - when using transport `streamable-http` you will also need to review the KDB-X MCP logs from your terminal

To enable Developer mode:

- Start Claude Desktop, click the menu in the upper-left corner > `Help` > `Troubleshooting` > `Enable Developer Mode` (Confirm any popups)
- Restart Claude Desktop, click the menu in the upper-left corner > `Developer` > Developer settings should now be populated

## Prompts/Resources/Tools

### Prompts

| Name | Purpose | Params | Return |
|-------------------|------------------------------------------|-------------------------------------------------|------------------------------------------------|
| kdbx_table_analysis | Generate a detailed analysis prompt for a specific table. | table_name: Name of the table to analyze<br> analysis_type (optional): Type of analysis options statistical, data_quality<br> sample_size (optional): Suggested sample size for data exploration | The generated table analysis prompt |

### Resources

| Name | URI | Purpose | Params |
|---------------------|---------------------|--------------------------------------------------------------------------------------------|--------------------------------------------|
| kdbx_describe_tables | kdbx://tables | Get comprehensive overview of all database tables with schema information and sample data. | None |
| kdbx_sql_query_guidance | file://guidance/kdbx-sql-queries | Sql query syntax guidance and examples to execute. | None |

### Tools

| Name | Purpose | Params | Return |
|-------------------|------------------------------------------|-------------------------------------------------|------------------------------------------------|
| kdbx_run_sql_query | Execute SQL SELECT against KDB-X database | query (str): SQL SELECT query string to execute | JSON object with query results (max 1000 rows) |

## Development

To add new tools:

1. Create a new Python file in src/mcp_server/tools/.
2. Implement your tool using the _template.py as a reference.
3. The tool will be auto-discovered and registered when the server starts.
4. Restart Claude Desktop to access your new tool.

To add new resources:

1. Create a new Python file in src/mcp_server/resources/.
2. Implement your resource using the _template.py as a reference.
3. The resource will be auto-discovered and registered when the server starts.
4. Restart Claude Desktop to access your new resource.

To add new prompts:

1. Create a new Python file in src/mcp_server/prompts/.
2. Implement your prompt using the _template.py as a reference.
3. The prompt will be auto-discovered and registered when the server starts.
4. Restart Claude Desktop to access your new prompt.

## Testing

The below tools can aid in the development, testing and debugging of new MCP tools, resource and prompts.

- [MCP Inspector](https://modelcontextprotocol.io/legacy/tools/inspector) is a interactive developer tool from Anthropic
- [Postman](https://learning.postman.com/docs/postman-ai-agent-builder/mcp-requests/create/) to create MCP requests and store in collections

## Troubleshooting

### KDB-X connection error

Ensure that your KDB-X database is online and accessible on the specified kdb host and port.

The default KDB-X endpoint is `localhost:5000`, but you can update as needed via section [Command line Parameters](#command-line-parameters).

#### KDB-X SQL interface error

The KDB-X MCP server communicates with the KDB-X service using its SQL interface.

If you get an error saying the SQL interface is not loaded. You can load it manually by running .s.init[]

```q
.s.init[]
```

### MCP Server port in use

If the MCP Server port is being used by another process you will need to specify a different port or stop the service that is using the port.

### Invalid transport

You can only specify `streamable-http`, `stdio.`

### Missing tools/resources

Review the Server logs for registration errors.

### Errors when interacting with a KDB-X database

Ensure the KDB-X resources are loaded, so Claude knows how to interact with the database.

- `kdbx_describe_tables`
- `kdbx_sql_query_guidance`

### UV Default Paths

| Platform | Default UV Path |
|----------|----------------|
| **macOS** | `~/.local/bin/uv` |
| **Linux** | `~/.local/bin/uv` |
| **Windows** | `%APPDATA%\Python\Scripts\uv.exe` |

### Claude Log Locations

| Platform | Path | Monitor Command |
|----------|------|-----------------|
| **macOS** | `~/Library/Logs/Claude/mcp*.log` | `tail -f ~/Library/Logs/Claude/mcp*.log` |
| **Windows** | `%APPDATA%\Claude\Logs\mcp*.log` | `Get-Content -Path "$env:APPDATA\Claude\Logs\mcp*.log" -Wait` |

### Official Claude Troubleshooting docs

For detailed troubleshooting, see [official Claude MCP docs](https://modelcontextprotocol.io/quickstart/user#troubleshooting).

### Claude limits

You may need to upgrade to a paid plan to avoid Claude usage errors like this:

> Claude hit the maximum length for this conversation. Please start a new conversation to continue chatting with Claude.

## Useful Resources

- [KDB-X documentation](https://docs.kx.com/public-preview/kdb-x/home.htm) for more general information about KDB-X development
- [KX Forum](https://forum.kx.com/) for community support
- [KX Slack](http://kx.com/slack) for support & feedback

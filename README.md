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
- [Security Considerations](#security-considerations)
- [Transport Options](#transport-options)
- [Command Line Tool](#command-line-tool)
- [Configure Embeddings](#configure-embeddings)
- [MCP Client Configuration](#mcp-client-configuration)
- [Prompts/Resources/Tools](#promptsresourcestools)
- [Development](#development)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Useful Resources](#useful-resources)

## Supported Environments

The following table shows the install options for supported Operating Systems:

| **Primary OS** | **KDB-X**       | **KDB+** | **MCP Server** | **UV/NPX** |
| -------------- | --------------- | -------- | -------------- | ---------- |
| **Mac**        | ✅ Local        | ✅ Local | ✅ Local       | ✅ Local   |
| **Linux**      | ✅ Local        | ✅ Local | ✅ Local       | ✅ Local   |
| **WSL**        | ✅ Local        | ✅ Local | ✅ Local       | ✅ Local   |
| **Windows**    | ⚠️ WSL          | ✅ Local | ⚠️ WSL         | ✅ Local   |
| **Windows**    | ⚠️ WSL          | ✅ Local | ✅ Local       | ✅ Local   |
| **Windows**    | ⚠️ Remote Linux | ✅ Local | ✅ Local       | ✅ Local   |

> The KDB-X MCP server can connect to one KDB Service - either KDB-X or KDB+, not both. \
> The chosen KDB Service needs to be listening on a host and port that is accessible to the KDB-X MCP server.

- **KDB-X**: Mac/Linux/WSL only (no native Windows support)
- **KDB+**: Windows/Mac/Linux/WSL
- **MCP Server**: UV required (Windows/Mac/Linux/WSL)
- **UV**: Required for running the MCP Server
- **NPX**: Required for streamable-http transport with Claude Desktop
- **Stdio transport**: Only works when your MCP Client and MCP Server are on same host

For details on MCP clients see [MCP Client Configuration](#mcp-client-configuration)

## Prerequisites

Before installing and running the KDB-X MCP Server, ensure you have met the following requirements:

- [Cloned this repo](#clone-the-repository)
- A `KDB-X/KDB+` Service listening on a host and port that will be accessible to the MCP Server
  - See examples - [KDB-X Setup](#kdb-x-setup) / [KDB+ Setup](#kdb-setup)
  - KDB-X can be installed by signing up to the [KDB-X public preview](https://developer.kx.com/products/kdb-x/install) - see [KDB-X documentation](https://docs.kx.com/public-preview/kdb-x/home.htm) for supporting information
  - Windows users can run the KDB-X MCP Server on Windows and connect to a local KDB-X database via WSL or remote KDB-X database running on Linux
  - Windows users can run a local KDB-X database by installing KDB-X on [WSL](https://learn.microsoft.com/en-us/windows/wsl/install), and use the default [streamable-http transport](#transport-options) when running the [KDB-X MCP Server](#run-the-server) - both share the same localhost network.
  - For details on KDB-X usage restrictions see [documentation](https://docs.kx.com/product/licensing/usage-restrictions.htm#kdb-x-personal-trial-download)
- [UV Installed](https://docs.astral.sh/uv/getting-started/installation/) for running the KDB-X MCP Server - available on Windows/Mac/Linux/WSL
- An [MCP Client](https://modelcontextprotocol.io/clients) installed - See [MCP Client Configuration](#mcp-client-configuration)
- [NPX](https://nodejs.org/en) is required to use `streamable-http` transport with Claude Desktop
  - `npx` may not be required if you are using a different MCP Client - consult the documentation of your chosen MCP Client
  - `npx` comes bundled with the [nodejs](https://nodejs.org/en) installer - available on Windows/Mac/Linux/WSL
  - See [example configuration with streamable-http](#example-configuration-with-streamable-http)

> Note: ⚠️ KDB-X public preview has recently been extended. If you have installed KDB-X prior to Sept 30th 2025, you will receive an email notification about this update. Please update to the latest [KDB-X](https://developer.kx.com/products/kdb-x/install) to ensure uninterrupted access, valid through 31st Dec 2025

## Quickstart

To demonstrate basic usage of the KDB-X MCP Server, using an empty KDB-X database, follow the quickstart steps below.

> Note: Ensure you have followed the necessary [prerequisites steps](#prerequisites)

1. Open a KDB-X service listening on a port.

   By default the KDB-X MCP server will connect to KDB-X service on port 5000 - [but this can be changed](#command-line-tool) via command line flags or environment variables.

   > Note: KDB-X is currently not supported on Windows - if you are using Windows we recommend running KDB-X on WSL as outlined in the [prerequisites steps](#prerequisites)

   ```bash
   q -p 5000
   ```

2. Load the ai and sql interfaces.

   ```q
   .ai:use`kx.ai
   .s.init[]
   ```

3. Add a dummy table e.g. `trade`.

   ```q
   rows:10000;
   trade:([]time:.z.d+asc rows?.z.t;sym:rows?`AAPL`GOOG`MSFT`TSLA`AMZN;price:rows?100f;size:rows?1000);
   ```

4. [Configure your MCP Client](#mcp-client-configuration) with your chosen transport.

5. [Start your MCP server](#mcp-server-installation).

   If you have configured your [MCP Client](#mcp-client-configuration) with [stdio transport](#transport-options), then this step is not required. Please move to the next step (Your MCP Client will manage starting the MCP Server for you).

   ```bash
   uv run mcp-server
   ```

6. Start your MCP Client and verify that the tools, prompts and resources section are visible. Consult your specific [MCP Client config](#mcp-client-configuration) for these details.

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

### Using AI tools with KDB-X

> Note: KDB+ users do not have access to similarity search tools

To enable the following tools

- "kdbx_similarity_search"
- "kdbx_hybrid_search"

with the KDB-X MCP Server you will need to:

- Be running **KDB-X version 0.1.2 or greater**.
- Have the ai-libs module loaded in your KDB-X session via:

```q
.ai:use`kx.ai
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

For Windows users with WSL installed - when using `streamable-http` transport, the MCP Server can run on either Windows or WSL. For both scenarios, the MCP Server will be available on the same shared localhost network. MCP Clients (running on Windows) will connect over `localhost`. So this repository can be cloned to either Windows or WSL. `uv` needs be installed on the same OS where the MCP Server will be running.

If you are using `stdio` on Windows, your MCP Client will manage starting and stopping the MCP server. So this repository will need to be cloned to the Windows filesystem. `uv` will need be installed on Windows.

```bash
uv run mcp-server
```

## Transport Options

For more info on the supported transports see official documentation

- [streamable-http](https://modelcontextprotocol.io/docs/concepts/transports#streamable-http)
- [stdio](https://modelcontextprotocol.io/docs/concepts/transports#standard-input%2Foutput-stdio)

> Note: We don't support [sse](https://modelcontextprotocol.io/docs/concepts/transports#server-sent-events-sse-deprecated) transport (server-sent events) as it has been deprecated since protocol version 2024-11-05.

## Security Considerations

To simplify getting started, we recommend running your MCP Client, KDB-X MCP server, and your KDB-X database on the same internal network.

### Encrypting Database Connections

If you require an encrypted connection between your KDB-X MCP server and your KDB-X database, you can enable enable TLS with `--db.tls=true`

This requires setting up your KDB-X database with TLS as a prerequisite:

- You can follow the [kdb+ SSL/TLS guide](https://code.kx.com/q/kb/ssl/) to setup TLS with your KDB-X database
- If you are using self signed certificates:
  - You will need to specify the location of your self signed CA cert
  - Set the `KX_SSL_CA_CERT_FILE` environment variable to point to the CA cert file that your KDB-X database is using
  - Alternatively, you can bypass certificate verification by setting `KX_SSL_VERIFY_SERVER=NO` for development and testing

### Encrypting MCP Client Connections

If you require an encrypted connection between your MCP Client and your KDB-X MCP server:

- The KDB-X MCP server uses streamable-http transport by default and starts a localhost server at 127.0.0.1:8000. We do not recommend exposing this externally.
- You can optionally setup an HTTPS proxy in front of your KDB-X MCP server such as [envoy](https://www.envoyproxy.io/) or [nginx](https://nginx.org/) for HTTPS termination
- When using stdio transport, this is not required as communication is through standard input/output streams on the same host

> Note: FastMCP v2 was evaluated for it's authentication features, but the KDB-X MCP Server will remain temporarily on v1 to preserve broad model compatibility until clients/models catch up, at which point we will transition.

## Command Line Tool

```bash
uv run mcp-server -h
usage: mcp-server [-h] [--mcp.server-name str] [--mcp.log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                  [--mcp.transport {stdio,streamable-http}] [--mcp.port int] [--mcp.host str] [--db.host str]
                  [--db.port int] [--db.username str] [--db.password SecretStr] [--db.tls bool] [--db.timeout int]
                  [--db.retry int] [--db.embedding-csv-path str] [--db.metric str] [--db.k int]

KDB-X MCP Server that enables interaction with KDB-X using natural language

options:
  -h, --help            show this help message and exit

mcp options:
  MCP server configuration and transport settings

  --mcp.server-name str
                        Name identifier for the MCP server instance [env: KDBX_MCP_SERVER_NAME] (default:
                        KDBX_MCP_Server)
  --mcp.log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Logging verbosity level [env: KDBX_MCP_LOG_LEVEL] (default: INFO)
  --mcp.transport {stdio,streamable-http}
                        Communication protocol: 'stdio' (pipes) or 'streamable-http' (HTTP server) [env:
                        KDBX_MCP_TRANSPORT] (default: streamable-http)
  --mcp.port int        HTTP server port - ignored when using stdio transport [env: KDBX_MCP_PORT] (default: 8000)
  --mcp.host str        HTTP server bind address - ignored when using stdio transport [env: KDBX_MCP_HOST] (default:
                        127.0.0.1)

db options:
  KDB-X database connection settings

  --db.host str         KDB-X server hostname or IP address [env: KDBX_DB_HOST] (default: 127.0.0.1)
  --db.port int         KDB-X server port number [env: KDBX_DB_PORT] (default: 5000)
  --db.username str     Username for KDB-X authentication [env: KDBX_DB_USERNAME] (default: )
  --db.password SecretStr
                        Password for KDB-X authentication [env: KDBX_DB_PASSWORD] (default: )
  --db.tls bool         Enable TLS for KDB-X connections. When using TLS you will need to set the environment variable
                        `KX_SSL_CA_CERT_FILE` that points to the certificate on your local filesystem that your KDB-X
                        server is using. For local development and testing you can set `KX_SSL_VERIFY_SERVER=NO` to
                        bypass this requirement [env: KDBX_DB_TLS] (default: False)
  --db.timeout int      Timeout in seconds for KDB-X connection attempts [env: KDBX_DB_TIMEOUT] (default: 1)
  --db.retry int        Number of connection retry attempts on failure [env: KDBX_DB_RETRY] (default: 2)
  --db.embedding-csv-path str
                        Path to embeddings csv [env: KDBX_DB_EMBEDDING_CSV_PATH] (default:
                        src/mcp_server/utils/embeddings.csv)
  --db.metric str       Distance metric used for vector similarity search (e.g., CS, L2, IP) [env: KDBX_DB_METRIC]
                        (default: CS)
  --db.k int            Default number of results to return from vector searches [env: KDBX_DB_K] (default: 5)
```

### CLI Configuration Options

The command line options are organized into two main categories:

- MCP Options - Configures the MCP server behavior and transport settings
- Database Options - Configures the KDB-X database connection settings

For details on each option, refer to the [help text](#command-line-tool)

### Configuration Methods

Configuration values are resolved in the following priority order:

1. **Command Line Arguments** - Highest priority
2. **Environment Variables** - Second priority
3. **.env File** - Third priority
4. **Default Values** - Default values defined in `settings.py`

### Environment Variables

Every command line option has a corresponding environment variable. For example:

- `--mcp.port 7001` ↔ `KDBX_MCP_PORT=7001`
- `--db.host localhost` ↔ `KDBX_DB_HOST=localhost`

> Note: `KDBX_DB_*` environment variables can be used when pointing to a KDB+ Service

### Example Usage

```bash
# Using defaults
uv run mcp-server

# Using a .env file
echo "KDBX_MCP_PORT=7001" >> .env
echo "KDBX_DB_RETRY=4" >> .env
uv run mcp-server

# Using environment variables
export KDBX_MCP_PORT=7001
export KDBX_DB_RETRY=4
uv run mcp-server

# Using command line arguments
uv run mcp-server \
    --mcp.port 7001 \
    --db.retry 4
```

## Configure Embeddings

Before starting the KDB-X MCP Server, you must configure embedding models for your tables if you wish to use Similarity Search.
The repository includes two ready-to-use embedding providers: OpenAI and SentenceTransformers.
You can customize these implementations as needed, or add your own provider by following the steps outlined below.

1. Update Dependencies - Add your required embedding providers to `pyproject.toml` dependencies section.

2. Set Environment Variables - Configure required API keys for your chosen embedding providers if necessary (for example, set the environment variable `OPENAI_API_KEY` to use OpenAI's API)

3. Add New Provider - The file `src/mcp_server/utils/embeddings.py` defines the base class `EmbeddingProvider` for all embedding providers.
   To add a new provider, create a class in the same file that extends this base class and implements all required abstract methods.
   You can use the existing implementations of OpenAI and SentenceTransformers in the same file as templates — simply copy and modify them to suit your needs. To register your provider, use the `@register_provider` decorator above your class definition. It is not compulsory for the registered provider name to follow the provider's Python package name.

4. Configure Table Embeddings - Update the embeddings configuration file at `src/mcp_server/utils/embeddings.csv` with your actual database and table names, embedding providers and models. The name you provide at `embeddings.csv` should match the registered provider name specified in file `embeddings.py`.

## MCP Client Configuration

The KDB-X MCP Server works with any MCP-compatible client.

## Configuration Guides

- [Claude Desktop](mcp-clients/claude-desktop.md) - macOS and Windows
- [GitHub Copilot in VSCode](mcp-clients/github-copilot-vscode.md) - macOS, Linux, Windows, and WSL

## Other MCP Clients

The KDB-X MCP Server is compatible with any MCP client that supports the Model Context Protocol. For a full list of compatible clients, see the [official MCP clients list](https://modelcontextprotocol.io/clients).

## Prompts/Resources/Tools

### Prompts

| Name                | Purpose                                                   | Params                                                                                                                                                                                                 | Return                              |
| ------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------- |
| kdbx_table_analysis | Generate a detailed analysis prompt for a specific table. | `table_name`: Name of the table to analyze<br> `analysis_type` (optional): Type of analysis options statistical, data_quality<br> `sample_size` (optional): Suggested sample size for data exploration | The generated table analysis prompt |

### Resources

| Name                    | URI                              | Purpose                                                                                    | Params |
| ----------------------- | -------------------------------- | ------------------------------------------------------------------------------------------ | ------ |
| kdbx_describe_tables    | kdbx://tables                    | Get comprehensive overview of all database tables with schema information and sample data. | None   |
| kdbx_sql_query_guidance | file://guidance/kdbx-sql-queries | Sql query syntax guidance and examples to execute.                                         | None   |

### Tools

| Name                   | Purpose                                                                                   | Params                                                                                                                                                          | Return                                         |
| ---------------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| kdbx_run_sql_query     | Execute SQL SELECT against KDB-X database                                                 | `query`: SQL SELECT query string to execute                                                                                                                     | JSON object with query results (max 1000 rows) |
| kdbx_similarity_search | Perform vector similarity search on a KDB-X table                                         | `table_name`: Name of the table to search <br> `query`: Text query to convert to vector and search <br> `n` (optional): Number of results to return             | Dictionary containing search result            |
| kdbx_hybrid_search     | Perform hybrid search combining vector similarity and sparse text search on a KDB-X table | table_name: Name of the table to search <br> query: Text query to convert to dense and sparse vectors for search <br> n (optional): Number of results to return | Dictionary containing search result            |

## Development

To add new tools:

1. Create a new Python file in src/mcp_server/tools/.
2. Implement your tool using the _template.py as a reference.
3. The tool will be auto-discovered and registered when the server starts.
4. Restart your MCP Client to access your new tool.

To add new resources:

1. Create a new Python file in src/mcp_server/resources/.
2. Implement your resource using the _template.py as a reference.
3. The resource will be auto-discovered and registered when the server starts.
4. Restart your MCP Client Desktop to access your new resource.

To add new prompts:

1. Create a new Python file in src/mcp_server/prompts/.
2. Implement your prompt using the _template.py as a reference.
3. The prompt will be auto-discovered and registered when the server starts.
4. Restart your MCP Client Desktop to access your new prompt.

## Testing

The below tools can aid in the development, testing and debugging of new MCP tools, resource and prompts.

- [MCP Inspector](https://modelcontextprotocol.io/legacy/tools/inspector) is a interactive developer tool from Anthropic
- [Postman](https://learning.postman.com/docs/postman-ai-agent-builder/mcp-requests/create/) to create MCP requests and store in collections

## Troubleshooting

This section covers common MCP server issues. For client-specific troubleshooting (configuration, connection, tools, prompts, resources), see:
- [Claude Desktop Troubleshooting](mcp-clients/claude-desktop.md#troubleshooting)
- [GitHub Copilot in VSCode Troubleshooting](mcp-clients/github-copilot-vscode.md#troubleshooting)

### Failed to import pykx

The KDB-X MCP Server requires a valid KDB-X license to operate.

If you see an error like "Failed to import pykx", verify the following:

- The `QLIC` environment variable is set and points to your license directory
- Your license directory contains a valid license file
- Your license has not expired

### KDB-X license expired

KDB-X public preview has recently been extended. If you have installed KDB-X prior to Sept 30th 2025, you will receive an email notification about this update. Please update to the latest [KDB-X](https://developer.kx.com/products/kdb-x/install) to ensure uninterrupted access, valid through 31st Dec 2025

### KDB-X connection error

Ensure that your KDB-X database is online and accessible on the specified kdb host and port.

The default KDB-X endpoint is `localhost:5000`, but you can update as needed via section [Command line Tool](#command-line-tool).

### KDB-X SQL interface error

The KDB-X MCP server communicates with the KDB-X service using its SQL interface.

If you get an error saying the SQL interface is not loaded. You can load it manually by running .s.init[]

```q
.s.init[]
```

### MCP Server port in use

If the MCP Server port is being used by another process you will need to specify a different port or stop the service that is using the port.

### Invalid transport

You can only specify `streamable-http` or `stdio.`

### Missing tools/resources

Review the Server logs for registration errors.
- Some tools may not be available for your version of KDB+ or KDB-X
- See section [Using similarity search tools with KDB-X](#using-similarity-search-tools-with-kdb-x) for more info.

### Errors when interacting with a KDB-X database

Ensure the KDB-X resources are loaded, so your MCP Client knows how to interact with the database.

- `kdbx_describe_tables`
- `kdbx_sql_query_guidance`

### UV Default Paths

| Platform    | Default UV Path                   |
| ----------- | --------------------------------- |
| **macOS**   | `~/.local/bin/uv`                 |
| **Linux**   | `~/.local/bin/uv`                 |
| **Windows** | `%APPDATA%\Python\Scripts\uv.exe` |


## Useful Resources

- [KDB-X documentation](https://docs.kx.com/public-preview/kdb-x/home.htm) for more general information about KDB-X development
- [KX Forum](https://forum.kx.com/) for community support
- [KX Slack](http://kx.com/slack) for support & feedback

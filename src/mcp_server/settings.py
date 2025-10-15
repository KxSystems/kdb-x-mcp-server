from typing import Literal
from pydantic import SecretStr, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class KDBConfig(BaseSettings):
    """KDB- X database connection settings"""

    model_config = SettingsConfigDict(
        env_prefix="KDBX_DB_",
        frozen=True,
        env_file='.env',
        extra="ignore",
    )

    host: str = Field(
        default="127.0.0.1",
        description="KDB-X server hostname or IP address [env: KDBX_DB_HOST]"
    )
    port: int = Field(
        default=5000,
        description="KDB-X server port number [env: KDBX_DB_PORT]"
    )
    username: str = Field(
        default="",
        description="Username for KDB-X authentication [env: KDBX_DB_USERNAME]"
    )
    password: SecretStr = Field(
        default=SecretStr(""),
        description="Password for KDB-X authentication [env: KDBX_DB_PASSWORD]"
    )
    tls: bool = Field(
        default=False,
        description="""Enable TLS for KDB-X connections.
        When using TLS you will need to set the environment variable `KX_SSL_CA_CERT_FILE` that points
        to the certificate on your local filesystem that your KDB-X server is using. For local development and testing
        you can set `KX_SSL_VERIFY_SERVER=NO` to bypass this requirement [env: KDBX_DB_TLS]"""
    )
    timeout: int = Field(
        default=1,
        description="Timeout in seconds for KDB-X connection attempts [env: KDBX_DB_TIMEOUT]"
    )
    retry: int = Field(
        default=2,
        description="Number of connection retry attempts on failure [env: KDBX_DB_RETRY]"
    )
    embedding_csv_path: str = Field(
        default="src/mcp_server/utils/embeddings.csv",
        description = "Path to embeddings csv [env: KDBX_DB_EMBEDDING_CSV_PATH]"
    )
    metric: str = Field(
        default="CS",
        description="Distance metric used for vector similarity search (e.g., CS, L2, IP) [env: KDBX_DB_METRIC]"
    )
    k: int = Field(
        default=5,
        description="Default number of results to return from vector searches [env: KDBX_DB_K]"
    )



class ServerConfig(BaseSettings):
    """MCP server configuration and transport settings."""

    model_config = SettingsConfigDict(
        env_prefix="KDBX_MCP_",
        env_file='.env',
        extra="ignore",
    )

    server_name: str = Field(
        default="KDBX_MCP_Server",
        description="Name identifier for the MCP server instance [env: KDBX_MCP_SERVER_NAME]"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging verbosity level [env: KDBX_MCP_LOG_LEVEL]"
    )
    transport: Literal["stdio", "streamable-http"] = Field(
        default="streamable-http",
        description="Communication protocol: 'stdio' (pipes) or 'streamable-http' (HTTP server) [env: KDBX_MCP_TRANSPORT]"
    )
    port: int = Field(
        default=8000,
        description="HTTP server port - ignored when using stdio transport [env: KDBX_MCP_PORT]"
    )
    host: str = Field(
        default="127.0.0.1",
        description="HTTP server bind address - ignored when using stdio transport [env: KDBX_MCP_HOST]"
    )


class AppSettings(BaseSettings):
    """KDB-X MCP Server that enables interaction with KDB-X using natural language"""

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_exit_on_error=True,
        cli_avoid_json=True,
        cli_prog_name="mcp-server",
        cli_kebab_case=True,
    )

    mcp: ServerConfig = Field(
        default_factory=ServerConfig,
        description="MCP server configuration and transport settings"
    )
    db: KDBConfig = Field(
        default_factory=KDBConfig,
        description="KDB-X database connection settings"
    )


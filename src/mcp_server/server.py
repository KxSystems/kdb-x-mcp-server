import sys
import logging
import socket
from mcp.server.fastmcp import FastMCP
from mcp_server.utils.logging import setup_logging
from mcp_server.settings import AppSettings
from mcp_server.tools import register_tools
from mcp_server.prompts import register_prompts
from mcp_server.resources import register_resources

# Global flag to track AI Libs availability
_ai_libs_available = False

def set_ai_libs_available(available: bool):
    global _ai_libs_available
    _ai_libs_available = available

def is_ai_libs_available() -> bool:
    return _ai_libs_available

class McpServer:

    def __init__(self, config: AppSettings):
        self.logger = logging.getLogger(__name__)

        self.db_config = config.db
        self.logger.info(f"KDBConfig: {self.db_config=}")

        self.mcp_config = config.mcp
        self.logger.info(f"ServerConfig: {self.mcp_config=}")

        # Initialize server
        self.mcp = FastMCP(
            self.mcp_config.server_name,
            port=self.mcp_config.port,
            host=self.mcp_config.host
        )

        self._check_port_availability()
        self._check_kdb_connection()
        self._register_tools()
        self._register_prompts()
        self._register_resources()
        if is_ai_libs_available():
            self._preload_embedding_models()

    def _check_port_availability(self):
        """Check if the configured mcp-port is available for HTTP transports."""
        if self.mcp_config.transport in ["streamable-http"]:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind((self.mcp_config.host, self.mcp_config.port))
                self.logger.info(
                    f"KDB-X MCP port availability check: SUCCESS - {self.mcp_config.host}:{self.mcp_config.port} is available"
                )
            except OSError:
                self.logger.error(f"KDB-X MCP port {self.mcp_config.port} is already in use on {self.mcp_config.host}")
                self.logger.error("Solutions:")
                self.logger.error(f"  - Try a different port: --mcp.port {self.mcp_config.port + 1}")
                self.logger.error(f"  - Stop the service using port {self.mcp_config.port}")
                sys.exit(1)

    def _check_kdb_connection(self):
        """Check if KDB-X service is reachable and accessible."""
        try:
            import pykx as kx
            from pykx.exceptions import QError
            conn = kx.SyncQConnection(
                host=self.db_config.host,
                port=self.db_config.port,
                username=self.db_config.username,
                password=self.db_config.password.get_secret_value(),
                timeout=self.db_config.timeout,
                tls=self.db_config.tls,

            )
            self.logger.info(f"KDB-X connectivity check with 'tls={self.db_config.tls}': SUCCESS - {self.db_config.host}:{self.db_config.port} is accessible")

            # check if sql interface is loaded on KDB-X service
            if not conn('@[{2< count .s};(::);{0b}]').py():
                self.logger.error("KDB-X SQL interface check: FAILED - KDB-X service does not have the SQL interface loaded. Load it by running .s.init[] in your KDB-X Session")
                sys.exit(1)
            else:
                self.logger.info("KDB-X SQL interface check: SUCCESS - SQL interface is loaded")

            # check if AI libs are loaded on KDB-X service
            ai_libs_available = conn('@[{2< count .ai};(::);{0b}]').py()
            if not ai_libs_available:
                self.logger.warning("KDB-X AI Libs check: NOT AVAILABLE - AI-powered tools (similarity_search, hybrid_search) will be disabled.")
                self.logger.warning("To enable AI tools, load the KDB-X AI libraries by running: \l ai-libs/init.q in your KDB-X Session and then restart the MCP server")
            else:
                self.logger.info("KDB-X AI Libs check: SUCCESS - AI Libs are loaded, AI tools will be available")

            set_ai_libs_available(ai_libs_available)
            conn.close()

        except QError as e:
            self.logger.error(f"KDB-X self.connectivity check with 'tls={self.db_config.tls}': FAILED - {self.db_config.host}:{self.db_config.port} ({e})")

            if "Connection refused" in str(e):
                self.logger.error(f"Verify KDB-X service is running and accessible on {self.db_config.host}:{self.db_config.port}")

            if "invalid username/password" in str(e):
                self.logger.error("Verify your KDBX_DB_USERNAME and KDBX_DB_PASSWORD are correct")

            self.logger.error("KDB-X MCP server cannot function without connection to a KDB-X database. Exiting...")
            sys.exit(1)

    def _register_tools(self):
        try:
            registered_tools = register_tools(self.mcp)
            self.logger.info(f"Successfully registered {len(registered_tools)} tools")

            for tool_name in registered_tools:
                self.logger.debug(f"Registered tool: {tool_name}")

        except Exception as e:
            self.logger.error(f"Failed to register tools: {e}")
            raise

    def _register_prompts(self):
        try:
            registered_prompts = register_prompts(self.mcp)
            self.logger.info(f"Successfully registered {len(registered_prompts)} prompts")

            for prompt_name in registered_prompts:
                self.logger.debug(f"Registered prompt: {prompt_name}")

        except Exception as e:
            self.logger.error(f"Failed to register prompts: {e}")
            raise

    def _register_resources(self):
        try:
            registered_resources = register_resources(self.mcp)
            self.logger.info(f"Successfully registered {len(registered_resources)} resources")

            for resource_name in registered_resources:
                self.logger.debug(f"Registered resource: {resource_name}")

        except Exception as e:
            self.logger.error(f"Failed to register resources: {e}")
            raise

    def _preload_embedding_models(self):
        """Preload embedding models to avoid first-request delays."""
        try:
            import asyncio
            from mcp_server.utils.embeddings import preload_models_from_config
            
            # Run the async preload function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(preload_models_from_config(self.db_config.embedding_csv_path))
                self.logger.info("Embedding models preloaded successfully")
            finally:
                loop.close()
                
        except Exception as e:
            self.logger.warning(f"Failed to preload embedding models: {e}")
            self.logger.warning("Models will be loaded on first use, which may cause delays")

    def run(self):
        """Start the MCP server."""
        try:
            self.logger.info(f"Starting {self.mcp_config.server_name} MCP Server with {self.mcp_config.transport} transport")
            self.mcp.run(transport=self.mcp_config.transport)
        except KeyboardInterrupt:
            self.logger.info("Server shutdown requested")
        except Exception as e:
            self.logger.error(f"Server error: {e}")
            raise
        finally:
            self.logger.info("Server stopped")


app_settings = AppSettings()


def main():
    """Main entry point for the KDB-X MCP Server."""

    # Setup logging with configured level
    setup_logging(app_settings.mcp.log_level)

    server = McpServer(app_settings)
    server.run()

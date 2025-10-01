
import os
import sys
import logging
import argparse
import socket
from typing import Optional

import pykx as kx
from pykx.exceptions import QError
from mcp.server.fastmcp import FastMCP
from mcp_server.utils.logging import setup_logging
from mcp_server.settings import ServerConfig, KDBConfig

try:
    from tools import register_tools
    from prompts import register_prompts
    from resources import register_resources
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from tools import register_tools
    from prompts import register_prompts
    from resources import register_resources

class MCPServerConfig:

    def __init__(self, kdbx_mcp_transport=None, kdbx_mcp_port=None, kdbx_host=None, kdbx_port=None, kdbx_timeout=None, kdbx_retry=None):

        self.logger = logging.getLogger(__name__)

        # Validate MCP configuration with CLI overrides
        mcp_kwargs = {}
        if kdbx_mcp_transport is not None:
            mcp_kwargs['transport'] = kdbx_mcp_transport
        if kdbx_mcp_port is not None:
            mcp_kwargs['port'] = kdbx_mcp_port
        self.settings = ServerConfig(**mcp_kwargs)

        # Validate KDB configuration with CLI overrides
        kdbx_kwargs = {}
        if kdbx_host is not None:
            kdbx_kwargs['host'] = kdbx_host
        if kdbx_port is not None:
            kdbx_kwargs['port'] = kdbx_port
        if kdbx_timeout is not None:
            kdbx_kwargs['timeout'] = kdbx_timeout
        if kdbx_retry is not None:
            kdbx_kwargs['retry'] = kdbx_retry
        self.kdbx_config = KDBConfig(**kdbx_kwargs)

        self.server_name = self.settings.server_name
        self.log_level = self.settings.log_level
        self.port = self.settings.port
        self.host = self.settings.host
        self.transport = self.settings.transport

# Global flag to track AI Libs availability
_ai_libs_available = False

def set_ai_libs_available(available: bool):
    global _ai_libs_available
    _ai_libs_available = available

def is_ai_libs_available() -> bool:
    return _ai_libs_available

class McpServer:

    def __init__(self, config: Optional[MCPServerConfig] = None):
        self.config = config or MCPServerConfig()
        self.logger = logging.getLogger(__name__)

        self.kdbx_config = self.config.kdbx_config
        self.logger.info(f"KDBConfig: {self.kdbx_config=}")

        self.server_config = self.config.settings
        self.logger.info(f"ServerConfig: {self.server_config=}")

        # Initialize server
        self.mcp = FastMCP(
            self.config.server_name,
            port=self.config.port,
            host=self.config.host
        )

        self._check_port_availability()
        self._check_kdb_connection()
        self._register_tools()
        self._register_prompts()
        self._register_resources()

    def _check_port_availability(self):
        """Check if the configured mcp-port is available for HTTP transports."""
        if self.config.transport in ['streamable-http']:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind((self.config.host, self.config.port))
                self.logger.info(f"KDB-X MCP port availability check: SUCCESS - {self.config.host}:{self.config.port} is available")
            except OSError:
                self.logger.error(f"KDB-X MCP port {self.config.port} is already in use on {self.config.host}")
                self.logger.error("Solutions:")
                self.logger.error(f"  - Try a different port: --kdbx-mcp-port {self.config.port + 1}")
                self.logger.error(f"  - Stop the service using port {self.config.port}")
                sys.exit(1)

    def _check_kdb_connection(self):
        """Check if KDB-X service is reachable and accessible."""
        try:
            conn = kx.SyncQConnection(
                host=self.kdbx_config.host,
                port=self.kdbx_config.port,
                username=self.kdbx_config.username,
                password=self.kdbx_config.password.get_secret_value(),
                timeout=5
            )
            self.logger.info(f"KDB-X connectivity check: SUCCESS - {self.kdbx_config.host}:{self.kdbx_config.port} is accessible")

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
            self.logger.error(f"KDB-X connectivity check: FAILED - {self.kdbx_config.host}:{self.kdbx_config.port} ({e})")

            if "Connection refused" in str(e):
                self.logger.error(f"Verify KDB-X service is running and accessible on {self.kdbx_config.host}:{self.kdbx_config.port}")

            if "invalid username/password" in str(e):
                self.logger.error("Verify your KDBX_USERNAME and KDBX_PASSWORD are correct")

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

    def run(self):
        """Start the MCP server."""
        try:
            self.logger.info(f"Starting {self.config.server_name} MCP Server with {self.config.transport} transport")
            self.mcp.run(transport=self.config.transport)
        except KeyboardInterrupt:
            self.logger.info("Server shutdown requested")
        except Exception as e:
            self.logger.error(f"Server error: {e}")
            raise
        finally:
            self.logger.info("Server stopped")


config: MCPServerConfig | None = None

def initialize_config(args):
    global config
    config = MCPServerConfig(kdbx_mcp_transport=args.kdbx_mcp_transport, kdbx_mcp_port=args.kdbx_mcp_port,
                            kdbx_host=args.kdbx_host, kdbx_port=args.kdbx_port,
                            kdbx_timeout=args.kdbx_timeout, kdbx_retry=args.kdbx_retry)

def parse_args():
    """Parse command line arguments for the MCP server."""
    parser = argparse.ArgumentParser(description='KDB-X MCP Server')

    # Transport mode flags (mutually exclusive)
    transport_group = parser.add_mutually_exclusive_group()
    transport_group.add_argument('--streamable-http', action='store_true',
                                help='Start the KDB-X MCP server with streamable HTTP transport (default)')
    transport_group.add_argument('--stdio', action='store_true',
                                help='Start the KDB-X MCP server with stdio transport')

    # Port configuration
    parser.add_argument('--kdbx-mcp-port', type=int,
                       help='Port number the KDB-X MCP server will listen on when using streamable-http transport (default 8000)')

    # KDB configuration
    parser.add_argument('--kdbx-host', type=str,
                       help='KDB-X host that the MCP server will connect to (default: localhost)')
    parser.add_argument('--kdbx-port', type=int,
                       help='KDB-X port that the MCP server will connect to (default: 5000)')
    parser.add_argument('--kdbx-timeout', type=int,
                       help='KDB-X connection timeout in seconds (default: 1)')
    parser.add_argument('--kdbx-retry', type=int,
                       help='KDB-X connection retry attempts (default: 2)')

    return parser.parse_args()


def main():
    args = parse_args()
     # Determine transport mode (only set if CLI flag is provided)
    args.kdbx_mcp_transport = None
    if args.streamable_http:
        args.kdbx_mcp_transport = 'streamable-http'
    elif args.stdio:
        args.kdbx_mcp_transport = 'stdio'

    # Create config with CLI overrides
    initialize_config(args)

    # Setup logging with configured level
    setup_logging(config.log_level)
    logger = logging.getLogger(__name__)

    # Warn if port is specified with stdio transport
    if config.transport == 'stdio' and args.kdbx_mcp_port:
        logger.warning("--kdbx-mcp-port is ignored with --stdio transport")

    server = McpServer(config)
    server.run()

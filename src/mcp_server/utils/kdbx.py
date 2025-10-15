import logging
from functools import lru_cache
import pykx as kx
from typing import Optional
from mcp_server.settings import KDBConfig
from mcp_server.server import app_settings

db_config = app_settings.db
logger = logging.getLogger(__name__)

def get_kdb_connection() -> kx.QConnection:

    try:
        conn = kdb_sync_connection(db_config)
        conn('') # check if conn is live for existing connection from cache
        return conn
    except Exception as e:
        if "Attempted to use a closed IPC connection" in str(e):
            logger.warning("KDB-X connection was closed. Reinitializing...")
            cleanup_kdb_connection()
            conn = kdb_sync_connection(db_config)
            conn('')
            return conn
        else:
            logger.error(f"Error in creating KDBX connection: {e}")
            raise

@lru_cache()
def kdb_sync_connection(config: Optional[KDBConfig] = None) -> kx.QConnection:
    if config is None:
        config = db_config

    logger.debug(f"KDBConfig: {config=}")
    logger.info(f"Connecting to KDB at {config.host}:{config.port}")
    retry = config.retry

    for attempt in range(1, retry + 1):
        try:
            conn = kx.SyncQConnection(
                host=config.host,
                port=config.port,
                username=config.username,
                password=config.password.get_secret_value(),
                timeout=config.timeout,
                reconnection_attempts=config.retry,
                tls=config.tls,
            )
            logger.info("Connected to Q/KDB-X")
            return conn
        except Exception as e:
            logger.warning(f"KDB-X connectivity attempt {attempt}/{retry} failed: {str(e)}")

    logger.error(f"Failed to connect to KDB")
    raise

def cleanup_kdb_connection():
    kdb_sync_connection.cache_clear()
    logger.info("KDBX connection cache cleared")

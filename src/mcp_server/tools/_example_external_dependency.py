"""
REFERENCE EXAMPLE — not a live tool.

Demonstrates wiring an external PyPI package (`humanize`) into a custom
tool, per docs/adding-external-dependencies.md. This file is prefixed
with `_` (like _template.py) specifically so the auto-discovery loader
in tools/__init__.py skips it — it will NOT register as a callable tool
on server startup. Copy this structure, without the leading underscore,
to build a real tool that uses an external dependency.

Setup (see docs/adding-external-dependencies.md for full detail):
    uv add humanize
"""

import logging
from typing import Dict, Any
import humanize  # the external dependency this example wires in
from mcp_server.utils.kdbx import get_kdb_connection

logger = logging.getLogger(__name__)


async def describe_table_size_impl(table_name: str) -> Dict[str, Any]:
    try:
        conn = get_kdb_connection()

        # A real kdb+ call, so this example is grounded in something
        # concrete rather than being a standalone toy import.
        row_count = int(conn(f"count {table_name}").py())
        byte_size = int(conn(f"-22!{table_name}").py())  # serialized size estimate

        # This is the actual external-dependency usage: humanize turns
        # raw numbers into a human-readable string, something the base
        # server has no built-in equivalent for.
        human_rows = humanize.intcomma(row_count)
        human_size = humanize.naturalsize(byte_size)

        return {
            "status": "success",
            "table_name": table_name,
            "row_count": row_count,
            "row_count_readable": human_rows,
            "estimated_size_bytes": byte_size,
            "estimated_size_readable": human_size,
        }

    except Exception as e:
        logger.error(f"Error in describe_table_size: {e}")
        return {"status": "error", "message": str(e)}


def register_tools(mcp_server):
    """
    Not called — this file is a reference example (see module docstring),
    excluded from auto-discovery by its leading underscore. Shown here
    only to illustrate the full pattern end-to-end.
    """
    @mcp_server.tool()
    async def describe_table_size(table_name: str) -> Dict[str, Any]:
        """
        Report a kdb+ table's row count and estimated size, in both raw
        and human-readable form (e.g. "1,500,000" rows, "1.5 MB").

        Args:
            table_name (str): Name of an in-memory kdb+ table.

        Returns:
            Dict[str, Any]: {"status", "table_name", "row_count",
            "row_count_readable", "estimated_size_bytes", "estimated_size_readable"}
        """
        return await describe_table_size_impl(table_name=table_name)

    return ['describe_table_size']

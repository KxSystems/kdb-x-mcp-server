import logging
from typing import List
from mcp.types import TextContent
from mcp_server.utils.kdbx import get_kdb_connection
from mcp_server.utils.format_utils import format_data_for_display
from mcp_server.utils.embeddings_helpers import get_embedding_config

logger = logging.getLogger(__name__)


async def kdbx_describe_table_impl(table: str) -> List[TextContent]:
    """
    Describe a specific KDB table with metadata and sample rows.

    Args:
        table: Name of the KDB table to describe

    Returns:
        List[TextContent]: Table description with metadata and sample data
    """
    try:
        conn = get_kdb_connection()

        total_records = conn('{count get x}', table).py()

        schema_data = conn.meta(table).py()
        partitioned_table = table in conn.Q.pt.py()

        output_lines = [
            f"\n  TABLE ANALYSIS: {table}",
            f"{'=' * 60}",
        ]

        output_lines.extend([
            f"\n Schema Information:",
            format_data_for_display(schema_data, table)
        ])

        if total_records > 0:
            preview_size = min(3, total_records)
            if not partitioned_table:
                preview_data = conn('{x sublist get y}', preview_size, table).py()
            else:
                preview_data = conn('{.Q.ind[get y;til x]}', preview_size, table).py()

            output_lines.extend([
                f"\n Data Preview ({preview_size} records):",
                format_data_for_display(preview_data, table)
            ])
        else:
            output_lines.append("\n Table is empty - no data to preview")

        final_output = "\n".join(output_lines)

        return [TextContent(type="text", text=final_output)]

    except Exception as error:
        logger.error(f"Failed to analyze table '{table}': {error}")
        error_output = f"\n TABLE ANALYSIS FAILED: {table}\n{'=' *60}\nError: {error}"
        return [TextContent(type="text", text=error_output)]


async def kdbx_describe_tables_impl() -> List[TextContent]:
    try:
        conn = get_kdb_connection()

        available_tables = conn.tables(None).py()
        
        # Filter out internal AI library index tables (*document, *stats, *token)
        available_tables = [
            table for table in available_tables 
            if not (table.endswith('document') or table.endswith('stats') or table.endswith('token'))
        ]

        if not available_tables:
            return [TextContent(
                type="text",
                text=" Database is empty - no tables found"
            )]

        overview_parts = [
            "  DATABASE SCHEMA OVERVIEW",
            "═" * 60,
            f" Found {len(available_tables)} table(s)\n"
        ]

        for table_name in available_tables:
            table_analysis = await kdbx_describe_table_impl(table_name)
            overview_parts.append(table_analysis[0].text)

        complete_overview = "\n".join(overview_parts)
        logger.debug(complete_overview)
        return [TextContent(type="text", text=complete_overview)]

    except Exception as error:
        logger.error(f"Database schema analysis failed: {error}")
        return [TextContent(
            type="text",
            text=f" DATABASE ANALYSIS ERROR\n{'═' * 60}\nFailed to analyze database schema: {error}"
        )]



def register_resources(mcp_server):
    @mcp_server.resource("kdbx://tables")
    async def kdbx_describe_tables() -> List[TextContent]:
        """
        Generate comprehensive KDB database schema overview with all KDB table details.

        Returns:
            List[TextContent]: Complete database analysis with all tables
        """
        return await kdbx_describe_tables_impl()

    return ['kdbx://tables']
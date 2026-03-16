import logging
import pykx as kx
import json
from typing import Dict, Any
from mcp_server.utils.kdbx import get_kdb_connection
from mcp_server.stats import tracker, track_size

logger = logging.getLogger(__name__)
MAX_ROWS_RETURNED = 1000

async def run_query_impl(sqlSelectQuery: str) -> Dict[str, Any]:
    try:
        dangerous_keywords = ['INSERT', 'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE']
        query_upper = sqlSelectQuery.upper().strip()

        for keyword in dangerous_keywords:
            if keyword in query_upper and not query_upper.startswith('SELECT'):
                raise ValueError(f"Query contains dangerous keyword: {keyword}")

        conn = get_kdb_connection()
        # below query gets kdbx table data back as json for correct conversion of different datatypes
        result = conn('{r:.s.e x;`rowCount`data!(count r;.j.j y sublist r)}', kx.CharVector(sqlSelectQuery), MAX_ROWS_RETURNED)
        total = int(result['rowCount'])
        if 0==total:
            return {"status": "success", "data": [], "message": "No rows returned"}
        # parse json result
        rows = json.loads(result['data'].py().decode('utf-8'))
        if total > MAX_ROWS_RETURNED:
            logger.info(f"Table has {total} rows. Query returned truncated data to {MAX_ROWS_RETURNED} rows.")
            return {
                "status": "success",
                "data": rows,
                "message": f"Showing first {MAX_ROWS_RETURNED} of {total} rows",
            }

        logger.info(f"Query returned {total} rows.")
        return {"status": "success", "data": rows}

    except Exception as e:
        logger.error(f"Query failed: {e}")
        if ".s.e" in str(e):
            logger.error(f"It looks like the SQL interface is not loaded. You can load it manually by running .s.init[]:")
            return {
                "status": "error",
                "error_type": "sql_interface_not_loaded",
                "message": "It looks like the SQL interface is not loaded in the KDB-X database. Please initialize it by running `.s.init[]` in your KDB-X session, or contact your system administrator.",
                "technical_details": str(e)
            }
        return {"status": "error", "message": str(e)}


def register_tools(mcp_server):
    @mcp_server.tool()
    @track_size(tracker, "kdbx_run_sql_query")
    async def kdbx_run_sql_query(query: str) -> Dict[str, Any]:
        """
        Execute a SQL query and return structured results only to be used on kdb and not on kdbai.

        This function processes SQL SELECT statements to retrieve data from the underlying
        database. It parses the query, executes it against the data source, and returns
        the results in a structured format suitable for further analysis or display.

        Use the kdbx_sql_query_guidance resource when creating queries


        Supported query types:
            - SELECT statements with column specifications
            - WHERE clauses for filtering
            - ORDER BY for result sorting
            - LIMIT for result pagination
            - Basic aggregation functions (COUNT, SUM, AVG, etc.)

        For query syntax and examples, see: file://guidance/kdbx-sql-queries

        Args:
            query (str): SQL SELECT query string to execute. Must be a valid SQL statement
                        following standard SQL syntax conventions.

        Returns:
            Dict[str, Any]: Query execution results.
        """
        return await run_query_impl(sqlSelectQuery=query)

    return ['kdbx_run_sql_query']

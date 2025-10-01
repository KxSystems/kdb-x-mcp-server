import logging
from typing import Optional, Dict, Any, List
from mcp_server.settings import KDBConfig
from mcp_server.utils.embeddings import get_provider
from mcp_server.utils.embeddings_helpers import get_embedding_config
import numpy as np
import pandas as pd
import logging
import pykx as kx
import json
from typing import Dict, Any
from mcp_server.utils.kdbx import get_kdb_connection

config = KDBConfig()
logger = logging.getLogger(__name__)


# Normalizes the result from the search operation
def normalize_result(df: Dict)-> Any:
    # serialize numpy ndarray type
    df = df.map(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    # convert timespan type (KDB time type)
    for col_name, col_type in df.dtypes.items():
        timespan_type = str(col_type).lower().startswith("timedelta")
        duration_type = str(col_type).lower().startswith("duration")
        if timespan_type or duration_type:
            df[col_name] = (pd.Timestamp("1970-01-01") + df[col_name]).dt.time
        # convert to dict
    return df.to_dict('records') if hasattr(df, 'to_dict') else df


async def kdbx_similarity_search_impl( table_name: str,
                                        query: str,
                                        n: Optional[int] = None) -> Dict[str, Any]:
    
    try:
        if n is None:
            n = config.k

        embeddings_column, embeddings_provider, embeddings_model, _, _ = get_embedding_config(table_name)
        
        dense_provider = get_provider(embeddings_provider)
        query_vector = await dense_provider.dense_embed(query, embeddings_model)

        # Build search parameters
        search_params = {
            "table" : table_name,
            "vcol"  : embeddings_column,
            "qvec"  : query_vector,
            "metric": config.metric,
            "n"     : int(n),
        }

        conn = get_kdb_connection()

        result = conn('''{[args]
                            c:args`vcol;
                            $[(args`table) in .Q.pt;
                                [
                                res:raze{[d;args;tbl;c]
                                    vecs:?[tbl;enlist (=;.Q.pf;d);0b;(enlist c)!enlist c]c;
                                    if[not count vecs; :()];
                                    res:.ai.flat.search[vecs;args`qvec;args`n;args`metric];
                                    res:res@\:iasc res[1];
                                    `dist xcols update dist:res[0] from ?[tbl;((=;.Q.pf;d);(in;`i;res[1]));0b;()]
                                }[;args;get args`table;c] each .Q.pv;
                                ![(args`n)#`dist xdesc res;();0b;enlist c]
                                ];
                                [
                                res:.ai.flat.search[?[args`table;();();c];args`qvec;args`n;args`metric];
                                ![(args`table) res[1];();0b;enlist c]
                                ]
                            ]}''', search_params)

        result = normalize_result(result.pd())

        return {
            "status": "success",
            "table": table_name,
            "recordsCount": len(result),
            "records": result
        }
    except Exception as e:
        logger.error(f"Error performing search on table {table_name}: {e}")
        return {
            "status": "error",
            "message": str(e),
            "table": table_name,
        } 


def register_tools(mcp_server):
    # Check if AI Libs are available
    try:
        from mcp_server.server import is_ai_libs_available
        
        if not is_ai_libs_available():
            logger.info("AI Libs not available - skipping...")
            return []
    except Exception as e:
        logger.warning(f"Could not check AI Libs availability: {e}. Skipping AI tools.")
        return []
    
    @mcp_server.tool()
    async def kdbx_similarity_search(table_name: str,
                            query: str,
                            n: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform vector similarity search on a KDB-X table.

        Args:
            table_name: Name of the table to search
            query: Text query to convert to vector and search
            n (Optional[int], optional): Number of results to return

        Returns:
            Dictionary containing search result.
        """
        results = await kdbx_similarity_search_impl(
            table_name,
            query, 
            n,
        )
        return results

    return ["kdbx_similarity_search"]
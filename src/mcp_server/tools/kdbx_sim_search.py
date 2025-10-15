import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from mcp_server.settings import KDBConfig
from mcp_server.utils.kdbx import get_kdb_connection
from mcp_server.utils.embeddings import get_provider
from mcp_server.utils.format_utils import normalize_search_result
from mcp_server.utils.embeddings_helpers import get_embedding_config


config = KDBConfig()
logger = logging.getLogger(__name__)


async def kdbx_similarity_search_impl(table_name: str,
                                        query: str,
                                        n: Optional[int] = None) -> Dict[str, Any]:
    
    try:
        if n is None:
            n = config.k

        embeddings_column, embeddings_provider, embeddings_model, _, _, _, _ = get_embedding_config(table_name)
        
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

        result = normalize_search_result(result.pd(), table_name)

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


async def kdbx_hybrid_search_impl(table_name: str,
                                    query: str,
                                    n: Optional[int] = None) -> Dict[str, Any]:
    
    try:
        if n is None:
            n = config.k

        embeddings_column, embeddings_provider, embeddings_model, _, sparse_index_name, sparse_tokenizer_provider, sparse_tokenizer_model = get_embedding_config(table_name)

        # Check if it has index 
        if sparse_index_name is None:
            logger.info(f"Error performing hybrid search on table {table_name}: Missing sparse index")
            return {
                    "status": "error",
                    "message": "The requested table does not have sparse index",
                    "table": table_name,
                } 

        dense_provider = get_provider(embeddings_provider)
        sparse_provider = dense_provider if embeddings_provider==sparse_tokenizer_provider else  get_provider(sparse_tokenizer_provider)
        query_vector = await dense_provider.dense_embed(query, embeddings_model)
        query_sparse = await sparse_provider.sparse_embed(query, sparse_tokenizer_model)

        # Build search parameters
        search_params = {
            "table"  : table_name,
            "vcol"   : embeddings_column,
            "dense"  : query_vector,
            "index"  : sparse_index_name,
            "sparse" : query_sparse,
            "metric" : config.metric,
            "n"      : int(n),
        }

        conn = get_kdb_connection()

        result = conn('''{[args]
                            c:args`vcol;
                            $[(args`table) in .Q.pt;
                                [
                                rdense:raze{[d;args;tbl;c]
                                    vecs:?[tbl;enlist (=;.Q.pf;d);0b;(enlist c)!enlist c]c;
                                    if[not count vecs; :()];
                                    res:.ai.flat.search[vecs;args`dense;args`n;args`metric];
                                    res:res@\:iasc res[1];
                                    `dist`id xcols update dist:res[0],id:((0,neg[1]_sums[.Q.cn[tbl]])@.Q.pv?/:d)+res[1] from ?[tbl;((=;.Q.pf;d);(in;`i;res[1]));0b;()]
                                }[;args;get args`table;c] each .Q.pv;
                                rdense:![(args`n)#`dist xdesc rdense;();0b;enlist c];
                                rsparse:.ai.bm25.psearch[args`index;args`sparse;args`n;1.25e;0.75e;.Q.pv];
                                if[not count rsparse[0];:()];
                                ids:(args`n)#.ai.hybrid.rrf[(rdense`id;rsparse[1]);60];
                                .Q.ind[get args`table;ids]
                                ];
                                [
                                rdense:.ai.flat.search[?[args`table;();();c];args`dense;args`n;args`metric];
                                rsparse:.ai.bm25.search[get (args`index);args`sparse;args`n;1.25e;0.75e];
                                if[not count rsparse[0];:()];
                                res:(args`n)#.ai.hybrid.rrf[(rdense[1];rsparse[1]);60];
                                ![(args`table) res;();0b;enlist c]
                                ]
                            ]}''', search_params)
        
        if hasattr(result, '__len__') and len(result) == 0:
            logger.info(f"Hybrid search on table {table_name} returned no results - sparse search may have found no matches")
            return {
                "status": "success",
                "table": table_name,
                "recordsCount": 0,
                "records": [],
                "message": "No results found - the sparse search returned no matches for the query"
            }

        result = normalize_search_result(result.pd(), table_name)

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
    
    @mcp_server.tool()
    async def kdbx_hybrid_search(table_name: str,
                                    query: str,
                                    n: Optional[int] = None) -> Dict[str, Any]:
        """
        Performs hybrid search on a KDB-X table by combining both vector and text(sparse) search.

        Args:
            table_name: Name of the table to search
            query: Text query to convert to sparse and dense vectors and search
            n (Optional[int], optional): Number of results to return

        Returns:
            Dictionary containing search result.
        """
        results = await kdbx_hybrid_search_impl(
            table_name,
            query, 
            n,
        )
        return results

    return ["kdbx_similarity_search", "kdbx_hybrid_search"]
import logging
import numpy as np
import pandas as pd
from typing import Any, Optional
from mcp_server.utils.embeddings_helpers import get_embedding_config

logger = logging.getLogger(__name__)


def remove_vector_columns(data: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Remove vector columns from DataFrame."""
    try:
        embedding_column, _, _, sparse_embedding_column, _, _, _ = get_embedding_config(table_name)
        
        # Remove dense embedding column if present
        if embedding_column and embedding_column in data.columns:
            data = data.drop(columns=[embedding_column])
            logger.debug(f"Removed embedding column '{embedding_column}' from results")
        
        # Remove sparse embedding column if present
        if sparse_embedding_column and sparse_embedding_column in data.columns:
            data = data.drop(columns=[sparse_embedding_column])
            logger.debug(f"Removed sparse embedding column '{sparse_embedding_column}' from results")
            
    except Exception as e:
        logger.debug(f"Could not get embedding config for table {table_name}: {e}")
    
    return data


def format_data_for_display(data: Any, table_name: Optional[str] = None) -> str:
    if isinstance(data, dict):
        if data and all(isinstance(v, (list, np.ndarray)) for v in data.values()):
            try:
                data = pd.DataFrame(data)
                if table_name:
                    data = remove_vector_columns(data, table_name)
            except Exception:
                pass
        elif data and all(isinstance(v, dict) for v in data.values()):
            lines = []
            for col_name, col_info in data.items():
                col_name_str = col_name[0] if isinstance(col_name, tuple) and len(col_name) == 1 else str(col_name)
                t_val = col_info.get('t', '')
                f_val = col_info.get('f', '')
                a_val = col_info.get('a', '')
                lines.append(f"  {col_name_str:20} | type={t_val:3} | f={f_val:5} | a={a_val}")
            return "\n".join(lines) if lines else str(data)
    
    if hasattr(data, 'drop') and hasattr(data, 'to_string'):
        if table_name:
            data = remove_vector_columns(data, table_name)
        
        return data.to_string(index=False, max_rows=10, max_cols=20)
    
    if hasattr(data, 'to_string') and not hasattr(data, 'drop'):
        return data.to_string()
    
    return str(data)


def normalize_search_result(df: pd.DataFrame, table_name: str) -> Any:
    # Serialize numpy ndarray type
    df = df.map(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    # Convert timespan type (KDB time type)
    for col_name, col_type in df.dtypes.items():
        timespan_type = str(col_type).lower().startswith("timedelta")
        duration_type = str(col_type).lower().startswith("duration")
        if timespan_type or duration_type:
            df[col_name] = (pd.Timestamp("1970-01-01") + df[col_name]).dt.time
    
    # Remove vector columns
    df = remove_vector_columns(df, table_name)
    
    # Convert to dict
    return df.to_dict('records') if hasattr(df, 'to_dict') else df
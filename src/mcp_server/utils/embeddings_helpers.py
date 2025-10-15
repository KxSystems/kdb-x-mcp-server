# from mcp_server.server import app_settings
from mcp_server.settings import KDBConfig
import pandas as pd
from functools import lru_cache
from typing import List, Optional
import logging

config = KDBConfig()
logger = logging.getLogger(__name__)

@lru_cache()
def get_csv_data(csv_path: str):
    df = pd.read_csv(csv_path)
    return df.where(pd.notna(df), None)


def get_embedding_config(
    table: str,
) -> List[Optional[str]]:
    """
    Get embedding or sparse tokenizer configuration for a specific database/table.

    Args:
        table: Table name to match

    Returns:
        List[Optional[str]]: [provider, model] based on config_type
    """
    # Filter rows where both database and table match
    df = get_csv_data(config.embedding_csv_path)
    matching_rows = df[df['table'] == table]


    if len(matching_rows) == 0:
        error_msg = f"No configuration found for table='{table}'"
        logger.error(error_msg)
        raise ValueError(error_msg)
    elif len(matching_rows) > 1:
        error_msg = f"Multiple configurations found for table='{table}'. Please ensure each table has only one configuration row."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Get the first matching row
    row = matching_rows.iloc[0]

    # Choose columns based on config_type flag
    return [row['embedding_column'],
            row['embedding_provider'],
            row['embedding_model'],
            row['sparse_embedding_column'],
            row['sparse_index_name'],
            row['sparse_tokenizer_provider'],
            row['sparse_tokenizer_model']]

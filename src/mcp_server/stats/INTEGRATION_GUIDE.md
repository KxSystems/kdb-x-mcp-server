# MCP Size Tracking Integration Guide

## Quick Start

### 1. Add to your MCP server

```python
from mcp.server.fastmcp import FastMCP
from mcp_size_tracker import SizeTracker, track_size
from typing import Dict, Any, Optional

# Initialize
mcp = FastMCP("KDB-X")
tracker = SizeTracker("kdbx_mcp_size_log.json")


@mcp.tool()
@track_size(tracker, "kdbx_run_sql_query")
async def kdbx_run_sql_query(query: str) -> Dict[str, Any]:
    """Execute a SQL SELECT query against KDB-X tables."""
    return await run_query_impl(sqlSelectQuery=query)


@mcp.tool()
@track_size(tracker, "kdbx_similarity_search")
async def kdbx_similarity_search(
    table_name: str,
    query: str,
    n: Optional[int] = None,
) -> Dict[str, Any]:
    """Perform vector similarity search on a KDB-X table."""
    return await kdbx_similarity_search_impl(table_name, query, n)
```

### 2. Alternative: Manual tracking

```python
@mcp.tool()
async def kdbx_run_sql_query(query: str) -> Dict[str, Any]:
    """Execute a SQL SELECT query against KDB-X tables."""
    import time
    start = time.time()

    result = await run_query_impl(sqlSelectQuery=query)

    duration_ms = (time.time() - start) * 1000
    tracker.log_call("kdbx_run_sql_query", {"query": query}, result, duration_ms)

    return result
```

## Viewing Statistics

### Basic usage

```bash
python view_stats.py
```

### Filter by date

```bash
python view_stats.py --since 2026-02-01
```

### Filter by tool

```bash
python view_stats.py --tool kdbx_run_sql_query
```

### Show detailed logs

```bash
python view_stats.py --detail
```

## Log File Format

The tracker creates `kdbx_mcp_size_log.json` with entries like:

```json
[
  {
    "timestamp": "2026-02-16T04:30:15.123456",
    "tool": "kdbx_run_sql_query",
    "query_size_mb": 0.00015,
    "response_size_mb": 2.45,
    "duration_ms": 234,
    "query_summary": {
      "query": "SELECT sym, price, size FROM trade WHERE date = 2026-02-16 LIMIT 100"
    }
  },
  {
    "timestamp": "2026-02-16T04:31:02.654321",
    "tool": "kdbx_similarity_search",
    "query_size_mb": 0.00008,
    "response_size_mb": 0.87,
    "duration_ms": 312,
    "query_summary": {
      "table_name": "news_embeddings",
      "query": "federal reserve interest rate decision",
      "n": 10
    }
  }
]
```

## Adding Custom Tracking

For non-MCP usage, use the tracker directly:

```python
from mcp_size_tracker import SizeTracker

tracker = SizeTracker("my_log.json")

query = "SELECT sym, price FROM trade WHERE date = 2026-02-16"
response = await kdbx_run_sql_query(query)

tracker.log_call("kdbx_run_sql_query", {"query": query}, response)

# Get stats
stats = tracker.get_stats(since_date="2026-02-16T00:00:00")
print(f"Total response data: {stats['total_response_mb']:.2f} MB")
```

## Monitoring Best Practices

1. **Rotate logs periodically** - Archive old logs to prevent file growth
2. **Set alerts** - Monitor for unusually large responses
3. **Track trends** - Compare daily/weekly totals
4. **Optimize queries** - Use aggregations and LIMIT clauses to reduce response sizes

## Log Rotation Script

```python
from pathlib import Path
from datetime import datetime, timedelta
import json
import shutil

def rotate_logs(log_file="mcp_size_log.json", keep_days=30):
    log_path = Path(log_file)
    if not log_path.exists():
        return

    # Archive current log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = log_path.parent / f"{log_path.stem}_{timestamp}.json"
    shutil.copy(log_path, archive_path)

    # Keep only recent entries
    with open(log_path) as f:
        logs = json.load(f)

    cutoff = datetime.now() - timedelta(days=keep_days)
    cutoff_str = cutoff.isoformat()

    recent = [l for l in logs if l["timestamp"] >= cutoff_str]

    with open(log_path, 'w') as f:
        json.dump(recent, f, indent=2)

    print(f"Archived {len(logs) - len(recent)} old entries")
```

## Example Output

```text
================================================================================
MCP API CALL SIZE SUMMARY
================================================================================
Total calls: 145
Date range: 2026-02-10 to 2026-02-16

BY TOOL:
--------------------------------------------------------------------------------

kdbx_run_sql_query:
  Calls:            128
  Total Query:      18.45 KB
  Total Response:   245.67 MB
  Avg Response:     1.92 MB
  Max Response:     15.34 MB
  Avg Duration:     156 ms

kdbx_similarity_search:
  Calls:            17
  Total Query:      2.31 KB
  Total Response:   8.12 MB
  Avg Response:     489.41 KB
  Max Response:     1.23 MB
  Avg Duration:     312 ms

================================================================================
OVERALL TOTALS:
  Total Query Data:    20.76 KB
  Total Response Data: 253.79 MB
  Combined:            253.81 MB
================================================================================
```

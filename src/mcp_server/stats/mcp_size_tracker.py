"""
Size tracking middleware for MCP server responses
Tracks get_data and get_meta API call sizes in MB
"""

import sys
import json
from datetime import datetime
from pathlib import Path
import logging

class SizeTracker:
    def __init__(self, log_file="kdbx_mcp_size_log.json"):
        self.log_file = Path(log_file)
        self.logger = logging.getLogger("kdbx_mcp_size_tracker")
        
    def get_size_mb(self, data):
        """Calculate size of data in MB"""
        if isinstance(data, str):
            size_bytes = len(data.encode('utf-8'))
        elif isinstance(data, (dict, list)):
            size_bytes = len(json.dumps(data, default=str).encode('utf-8'))
        else:
            size_bytes = sys.getsizeof(data)
        return size_bytes / (1024 * 1024)  # Convert to MB
    
    def log_call(self, tool_name, query, response, duration_ms=None):
        """Log API call with size metrics"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "tool": tool_name,
            "query_size_mb": self.get_size_mb(query),
            "response_size_mb": self.get_size_mb(response),
            "duration_ms": duration_ms,
            "query_summary": self._summarize_query(query)
        }
        
        # Append to log file
        try:
            if self.log_file.exists():
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(entry)
            
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to log size: {e}")
        
        return entry
    
    def _summarize_query(self, query):
        """Create a brief summary of the query parameters"""
        if isinstance(query, dict):
            # Create a compact summary of all parameters
            summary = {}
            for key, value in query.items():
                # Truncate long string values
                if isinstance(value, str) and len(value) > 100:
                    summary[key] = value[:100] + "..."
                # Summarize lists/dicts
                elif isinstance(value, (list, dict)):
                    summary[key] = f"<{type(value).__name__} len={len(value)}>"
                else:
                    summary[key] = value
            return summary
        # Fallback for non-dict queries
        return str(query)[:100]
    
    def get_stats(self, since_date=None):
        """Get aggregated statistics"""
        if not self.log_file.exists():
            return {"total_calls": 0, "total_response_mb": 0}
        
        with open(self.log_file, 'r') as f:
            logs = json.load(f)
        
        if since_date:
            logs = [l for l in logs if l["timestamp"] >= since_date]
        
        stats = {
            "total_calls": len(logs),
            "total_query_mb": sum(l["query_size_mb"] for l in logs),
            "total_response_mb": sum(l["response_size_mb"] for l in logs),
            "by_tool": {}
        }
        
        for log in logs:
            tool = log["tool"]
            if tool not in stats["by_tool"]:
                stats["by_tool"][tool] = {
                    "calls": 0,
                    "total_response_mb": 0,
                    "avg_response_mb": 0
                }
            stats["by_tool"][tool]["calls"] += 1
            stats["by_tool"][tool]["total_response_mb"] += log["response_size_mb"]
        
        # Calculate averages
        for tool, data in stats["by_tool"].items():
            data["avg_response_mb"] = data["total_response_mb"] / data["calls"]
        
        return stats


# Decorator for tracking tool calls
def track_size(tracker, tool_name):
    """Decorator to track size of tool calls"""
    from functools import wraps

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            import time
            import inspect
            start = time.time()

            # Capture all input parameters (more generic than just 'query')
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            input_params = dict(bound_args.arguments)

            # Execute the actual tool
            response = await func(*args, **kwargs)

            duration_ms = (time.time() - start) * 1000

            # Log the call
            tracker.log_call(tool_name, input_params, response, duration_ms)

            return response
        return wrapper
    return decorator


# Example integration with your MCP server
"""
from mcp.server.fastmcp import FastMCP
from mcp_size_tracker import SizeTracker, track_size

mcp = FastMCP("KDB-X")
tracker = SizeTracker("kdbx_mcp_size_log.json")

@mcp.tool()
@track_size(tracker, "kdbx_run_sql_query")
async def kdbx_run_sql_query(query: str) -> str:
    # Your existing implementation
    result = kx_client.getData(query)
    return result
"""

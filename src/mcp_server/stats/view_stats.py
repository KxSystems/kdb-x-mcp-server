#!/usr/bin/env python3
"""
View MCP size tracking statistics
Usage: python view_stats.py [--since YYYY-MM-DD] [--tool TOOL_NAME]
"""

import json
import argparse
from pathlib import Path

def format_mb(mb):
    """Format MB size"""
    if mb < 1:
        return f"{mb*1024:.2f} KB"
    return f"{mb:.2f} MB"

def main():
    parser = argparse.ArgumentParser(description="View MCP size tracking stats")
    parser.add_argument("--log-file", default="mcp_size_log.json", help="Log file path")
    parser.add_argument("--since", help="Filter since date (YYYY-MM-DD)")
    parser.add_argument("--tool", help="Filter by tool name")
    parser.add_argument("--detail", action="store_true", help="Show detailed logs")
    args = parser.parse_args()
    
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"No log file found at {log_path}")
        return
    
    with open(log_path) as f:
        logs = json.load(f)
    
    # Apply filters
    if args.since:
        since_dt = f"{args.since}T00:00:00"
        logs = [l for l in logs if l["timestamp"] >= since_dt]
    
    if args.tool:
        logs = [l for l in logs if l["tool"] == args.tool]
    
    if not logs:
        print("No matching logs found")
        return
    
    # Summary stats
    print("=" * 80)
    print("MCP API CALL SIZE SUMMARY")
    print("=" * 80)
    print(f"Total calls: {len(logs)}")
    print(f"Date range: {logs[0]['timestamp'][:10]} to {logs[-1]['timestamp'][:10]}")
    print()
    
    # Aggregate by tool
    by_tool = {}
    for log in logs:
        tool = log["tool"]
        if tool not in by_tool:
            by_tool[tool] = {
                "calls": 0,
                "total_query_mb": 0,
                "total_response_mb": 0,
                "max_response_mb": 0,
                "total_duration_ms": 0
            }
        by_tool[tool]["calls"] += 1
        by_tool[tool]["total_query_mb"] += log["query_size_mb"]
        by_tool[tool]["total_response_mb"] += log["response_size_mb"]
        by_tool[tool]["max_response_mb"] = max(
            by_tool[tool]["max_response_mb"], 
            log["response_size_mb"]
        )
        if log.get("duration_ms"):
            by_tool[tool]["total_duration_ms"] += log["duration_ms"]
    
    # Print per-tool stats
    print("BY TOOL:")
    print("-" * 80)
    for tool, stats in sorted(by_tool.items()):
        avg_response = stats["total_response_mb"] / stats["calls"]
        avg_duration = stats["total_duration_ms"] / stats["calls"] if stats["total_duration_ms"] > 0 else 0
        
        print(f"\n{tool}:")
        print(f"  Calls:            {stats['calls']}")
        print(f"  Total Query:      {format_mb(stats['total_query_mb'])}")
        print(f"  Total Response:   {format_mb(stats['total_response_mb'])}")
        print(f"  Avg Response:     {format_mb(avg_response)}")
        print(f"  Max Response:     {format_mb(stats['max_response_mb'])}")
        if avg_duration > 0:
            print(f"  Avg Duration:     {avg_duration:.0f} ms")
    
    # Overall totals
    total_query = sum(s["total_query_mb"] for s in by_tool.values())
    total_response = sum(s["total_response_mb"] for s in by_tool.values())
    
    print("\n" + "=" * 80)
    print("OVERALL TOTALS:")
    print(f"  Total Query Data:    {format_mb(total_query)}")
    print(f"  Total Response Data: {format_mb(total_response)}")
    print(f"  Combined:            {format_mb(total_query + total_response)}")
    print("=" * 80)
    
    # Detailed logs if requested
    if args.detail:
        print("\nDETAILED LOGS:")
        print("-" * 80)
        for log in logs[-20:]:  # Last 20 entries
            print(f"\n{log['timestamp']}")
            print(f"  Tool:     {log['tool']}")
            print(f"  Query:    {format_mb(log['query_size_mb'])}")
            print(f"  Response: {format_mb(log['response_size_mb'])}")
            if log.get('query_summary'):
                print(f"  Summary:  {log['query_summary']}")
            if log.get('duration_ms'):
                print(f"  Duration: {log['duration_ms']:.0f} ms")

if __name__ == "__main__":
    main()

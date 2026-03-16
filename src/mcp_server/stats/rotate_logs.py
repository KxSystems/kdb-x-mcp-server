#!/usr/bin/env python3 
from pathlib import Path
from datetime import datetime, timedelta
import json
import argparse
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

def main():
    parser = argparse.ArgumentParser(description="Trim MCP size tracking stats log")
    parser.add_argument("--log-file", default="mcp_size_log.json", help="Log file path")
    parser.add_argument("--keep-days", default=7, type=int, help="Number of days to keep")
    args = parser.parse_args()

    rotate_logs(args.log_file, args.keep_days)

if __name__ == "__main__":
    main()
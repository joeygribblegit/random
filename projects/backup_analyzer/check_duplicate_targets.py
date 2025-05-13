#!/usr/bin/env python3

import os
import json
from pathlib import Path
import logging
from collections import defaultdict
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TargetPathChecker:
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.target_to_sources = defaultdict(list)  # Maps target path to list of source paths
        self.total_targets = 0
        
    def check_log_files(self):
        """Check all log files for duplicate target paths."""
        # Get all log files in the log directory
        log_files = sorted(self.log_dir.glob("copies_log_*.json"))
        
        if not log_files:
            logger.error(f"No log files found in {self.log_dir}")
            return
            
        logger.info(f"Found {len(log_files)} log files to check")
        
        # Process each log file
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
                
                # Process each copy entry
                for copy in log_data.get('copies', []):
                    target = copy.get('target')
                    source = copy.get('source')
                    if target and source:
                        self.target_to_sources[target].append(source)
                        self.total_targets += 1
                        
            except Exception as e:
                logger.error(f"Error processing log file {log_file}: {e}")
                continue
        
        # Find and report duplicates
        duplicates = {target: sources for target, sources in self.target_to_sources.items() 
                     if len(sources) > 1}
        
        logger.info(f"\nTotal target paths checked: {self.total_targets}")
        logger.info(f"Unique target paths: {len(self.target_to_sources)}")
        
        if duplicates:
            logger.info(f"Found {len(duplicates)} duplicate target paths:")
            for target, sources in duplicates.items():
                logger.info(f"\nTarget: {target}")
                logger.info("Source files:")
                for source in sources:
                    logger.info(f"  - {source}")
        else:
            logger.info("No duplicate target paths found")

def main():
    parser = argparse.ArgumentParser(description='Check for duplicate target paths in backup logs')
    parser.add_argument('--log-dir', default='/Users/jgribble/code/projects/backup_analyzer/logs',
                      help='Directory containing log files')
    
    args = parser.parse_args()
    
    checker = TargetPathChecker(args.log_dir)
    checker.check_log_files()

if __name__ == '__main__':
    main() 
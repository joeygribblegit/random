#!/usr/bin/env python3

import json
import os
from pathlib import Path
import logging
import argparse
from datetime import datetime
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FileRemover:
    def __init__(self, log_dir: str, target_dir: str, dry_run: bool = True):
        self.log_dir = Path(log_dir)
        self.target_dir = Path(target_dir)
        self.dry_run = dry_run
        self.removed_files = []
        self.failed_removals = []
        
        # Create removal log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.removal_log = self.log_dir / f"removal_log_{timestamp}.json"
        
    def _deduplicate_files(self, files: list) -> list:
        """Deduplicate files based on target path, keeping the most recent copy."""
        # Create a dictionary to store the most recent file info for each target path
        target_to_file = {}
        
        for file_info in files:
            target = file_info['target']
            timestamp = file_info.get('timestamp', '')
            
            # If we haven't seen this target before, or if this copy is more recent
            if target not in target_to_file or timestamp > target_to_file[target]['timestamp']:
                target_to_file[target] = file_info
        
        # Convert back to list
        deduplicated = list(target_to_file.values())
        
        # Log deduplication results
        duplicates_removed = len(files) - len(deduplicated)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate file entries")
            
        return deduplicated
        
    def find_files_to_remove(self, pattern: str) -> list:
        """Find files in log files that match the pattern."""
        files_to_remove = []
        
        # Get all log files
        log_files = sorted(self.log_dir.glob("copies_log_*.json"))
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
                
                # Check each copy in the log
                for copy in log_data.get('copies', []):
                    source = copy.get('source', '')
                    target = copy.get('target', '')
                    
                    # If source path contains pattern, add target to removal list
                    if pattern in source:
                        files_to_remove.append({
                            'source': source,
                            'target': target,
                            'size': copy.get('size_bytes', 0),
                            'timestamp': copy.get('timestamp', '')
                        })
                        
            except Exception as e:
                logger.error(f"Error reading log file {log_file}: {e}")
                continue
        
        # Deduplicate files before returning
        return self._deduplicate_files(files_to_remove)
    
    def move_files_to_folder(self, files_to_move: list, destination_folder: str):
        """Move files to a specified destination folder."""
        total_size = 0
        total_files = len(files_to_move)
        destination = Path(destination_folder)
        
        logger.info(f"Found {total_files} files to move to {destination}")
        
        # Create destination folder if it doesn't exist
        if not self.dry_run:
            try:
                destination.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Error creating destination folder: {e}")
                return
        
        for file_info in files_to_move:
            source_path = Path(file_info['target'])
            # Create new path in destination folder
            dest_path = destination / source_path.name
            
            try:
                if not source_path.exists():
                    logger.warning(f"Source file does not exist: {source_path}")
                    self.failed_removals.append({
                        **file_info,
                        'error': 'File not found'
                    })
                    continue
                
                if not self.dry_run:
                    # Move the file
                    shutil.move(str(source_path), str(dest_path))
                    total_size += file_info['size']
                    self.removed_files.append({
                        **file_info,
                        'new_location': str(dest_path)
                    })
                    logger.info(f"Moved: {source_path} -> {dest_path}")
                else:
                    # Just log what would be moved
                    total_size += file_info['size']
                    self.removed_files.append({
                        **file_info,
                        'new_location': str(dest_path)
                    })
                    logger.info(f"Would move: {source_path} -> {dest_path}")
                    
            except Exception as e:
                logger.error(f"Error moving {source_path}: {e}")
                self.failed_removals.append({
                    **file_info,
                    'error': str(e)
                })
        
        # Save removal log
        self._save_removal_log()
        
        # Print summary
        logger.info("\nMove Summary:")
        logger.info(f"Total files: {total_files}")
        logger.info(f"Successfully moved: {len(self.removed_files)}")
        logger.info(f"Failed to move: {len(self.failed_removals)}")
        logger.info(f"Total size: {self._format_size(total_size)}")
        if not self.dry_run:
            logger.info(f"Move log saved to: {self.removal_log}")
    
    def remove_files(self, files_to_remove: list):
        """Remove files from target directory."""
        total_size = 0
        total_files = len(files_to_remove)
        
        logger.info(f"Found {total_files} files to remove")
        
        for file_info in files_to_remove:
            target_path = Path(file_info['target'])
            
            try:
                if not target_path.exists():
                    logger.warning(f"Target file does not exist: {target_path}")
                    self.failed_removals.append({
                        **file_info,
                        'error': 'File not found'
                    })
                    continue
                
                if not self.dry_run:
                    # Remove the file
                    target_path.unlink()
                    total_size += file_info['size']
                    self.removed_files.append(file_info)
                    logger.info(f"Removed: {target_path}")
                else:
                    # Just log what would be removed
                    total_size += file_info['size']
                    self.removed_files.append(file_info)
                    logger.info(f"Would remove: {target_path}")
                    
            except Exception as e:
                logger.error(f"Error removing {target_path}: {e}")
                self.failed_removals.append({
                    **file_info,
                    'error': str(e)
                })
        
        # Save removal log
        self._save_removal_log()
        
        # Print summary
        logger.info("\nRemoval Summary:")
        logger.info(f"Total files: {total_files}")
        logger.info(f"Successfully removed: {len(self.removed_files)}")
        logger.info(f"Failed to remove: {len(self.failed_removals)}")
        logger.info(f"Total size: {self._format_size(total_size)}")
        if not self.dry_run:
            logger.info(f"Removal log saved to: {self.removal_log}")
    
    def _save_removal_log(self):
        """Save removal log to file."""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'dry_run': self.dry_run,
            'removed_files': self.removed_files,
            'failed_removals': self.failed_removals
        }
        
        try:
            with open(self.removal_log, 'w') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving removal log: {e}")
    
    def _format_size(self, size_bytes: int) -> str:
        """Convert bytes to human readable format."""
        for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PiB"

def main():
    parser = argparse.ArgumentParser(description='Remove or move files from organized backup based on log files')
    parser.add_argument('pattern', help='Pattern to match in source paths (e.g., "From School/yb/")')
    parser.add_argument('--log-dir', default='/Users/jgribble/code/projects/backup_analyzer/logs',
                      help='Directory containing log files')
    parser.add_argument('--target-dir', default='/Volumes/Samsung T7/organized_backup',
                      help='Target directory containing organized files')
    parser.add_argument('--dry-run', action='store_true', default=False,
                      help='Run in dry-run mode (default: True)')
    parser.add_argument('--move-to', help='Move matching files to this directory instead of removing them')
    
    args = parser.parse_args()
    
    remover = FileRemover(args.log_dir, args.target_dir, dry_run=args.dry_run)
    files_to_process = remover.find_files_to_remove(args.pattern)
    
    if args.move_to:
        remover.move_files_to_folder(files_to_process, args.move_to)
    else:
        remover.remove_files(files_to_process)

if __name__ == "__main__":
    main() 
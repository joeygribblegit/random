#!/usr/bin/env python3

import os
import hashlib
from collections import defaultdict
from pathlib import Path
import mimetypes
from datetime import datetime
import json
from typing import Dict, List, Set, Tuple
import argparse
import multiprocessing as mp
from tqdm import tqdm
import time
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
import psutil
import threading
import logging
import mmap
from itertools import islice
import signal

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for optimization
CHUNK_SIZE = 8192  # 8KB chunks for hashing
LARGE_FILE_THRESHOLD = 100 * 1024 * 1024  # 100MB
MAX_WORKERS = min(mp.cpu_count(), 16)  # Use more cores but cap at 16
BATCH_SIZE = 1000  # Process files in batches

class BackupAnalyzer:
    def __init__(self, root_dir: str, dry_run: bool = True, cache_file: str = None, force: bool = False, skip_large: bool = False, test_mode: bool = False):
        self.root_dir = Path(root_dir)
        self.dry_run = dry_run
        self.force = force
        self.skip_large = skip_large
        self.test_mode = test_mode
        self.file_hashes = defaultdict(list)
        self.file_types = defaultdict(list)
        self.total_size = 0
        self.file_count = 0
        self.processed_files = set()
        self.should_exit = False
        self.start_time = None
        
        # Define patterns to ignore
        self.ignored_patterns = {
            '.git', '.DS_Store', 'Thumbs.db', 
            '.cache', '__pycache__', 'node_modules',
            '.env', '.venv', 'venv'
        }
        
        # Set up output directory
        self.output_dir = Path("/Users/jgribble/code/projects/backup_analyzer")
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up cache file in the output directory if not specified
        if cache_file:
            self.cache_file = Path(cache_file)
        else:
            # Use the output directory for cache
            self.cache_file = self.output_dir / "cache.pkl"
        
        logger.info(f"Using cache file: {self.cache_file}")

    def handle_interrupt(self, signum, frame):
        """Handle interrupt signal (Ctrl+C/Command+C) gracefully."""
        logger.info("\nReceived interrupt signal. Initiating graceful shutdown...")
        self.should_exit = True
        if self.executor:
            logger.info("Shutting down process pool...")
            self.executor.shutdown(wait=False)
        self.save_cache()
        self.print_final_summary()
        sys.exit(0)

    def print_final_summary(self):
        """Print a summary of what was processed before exiting."""
        if self.start_time:
            elapsed = time.time() - self.start_time
            files_per_second = self.file_count / elapsed if elapsed > 0 else 0
            logger.info("\n=== Final Summary ===")
            logger.info(f"Processed {self.file_count} files")
            logger.info(f"Total size: {self._format_size(self.total_size)}")
            logger.info(f"Processing speed: {files_per_second:.1f} files/sec")
            logger.info(f"Time elapsed: {elapsed:.1f} seconds")
            logger.info("Progress has been saved to cache.")

    def load_cache(self) -> bool:
        """Load cached analysis results if they exist and are still valid."""
        if self.force:
            logger.info("Force flag set, ignoring cache")
            return False
            
        if not os.path.exists(self.cache_file):
            return False
            
        try:
            with open(self.cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                
            # Check if the root directory modification time has changed
            current_mtime = os.path.getmtime(self.root_dir)
            if cached_data.get('last_modified') != current_mtime:
                return False
                
            self.file_hashes = cached_data['file_hashes']
            self.file_types = cached_data['file_types']
            self.total_size = cached_data['total_size']
            self.file_count = cached_data['file_count']
            self.processed_files = cached_data.get('processed_files', set())
            self.last_modified = current_mtime
            return True
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return False

    def save_cache(self):
        """Save analysis results to cache file."""
        try:
            cache_data = {
                'file_hashes': self.file_hashes,
                'file_types': self.file_types,
                'total_size': self.total_size,
                'file_count': self.file_count,
                'last_modified': os.path.getmtime(self.root_dir),
                'processed_files': self.processed_files
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file using optimized reading strategy."""
        sha256_hash = hashlib.sha256()
        try:
            file_size = file_path.stat().st_size
            
            # For large files, use memory-mapped reading
            if file_size > LARGE_FILE_THRESHOLD:
                with open(file_path, 'rb') as f:
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                        sha256_hash.update(mm)
            else:
                # For smaller files, use chunked reading
                with open(file_path, "rb") as f:
                    while chunk := f.read(CHUNK_SIZE):
                        sha256_hash.update(chunk)
                        
            return sha256_hash.hexdigest()
        except (PermissionError, FileNotFoundError):
            # Don't log these common errors - they're expected for some files
            return ""
        except Exception as e:
            # Only log unexpected errors
            logger.debug(f"Unexpected error reading {file_path}: {e}")
            return ""

    def get_file_type(self, file_path: Path) -> str:
        """Determine file type based on extension and mime type."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            return mime_type.split('/')[0]  # Returns 'image', 'video', 'text', etc.
        return "unknown"

    def should_ignore(self, path: Path) -> bool:
        """Check if the path should be ignored."""
        return any(pattern in str(path) for pattern in self.ignored_patterns)

    def process_files_batch(self, file_paths: List[Path]) -> List[Tuple[str, str, int, str]]:
        """Process a batch of files and return their information."""
        results = []
        logger.debug(f"Starting batch processing of {len(file_paths)} files")
        for file_path in file_paths:
            try:
                logger.debug(f"Processing file: {file_path}")
                stats = file_path.stat()
                file_hash = self.calculate_file_hash(file_path)
                file_type = self.get_file_type(file_path)
                logger.debug(f"Successfully processed {file_path}: hash={file_hash[:8]}..., type={file_type}, size={stats.st_size}")
                results.append((file_hash, file_type, stats.st_size, str(file_path)))
            except (PermissionError, FileNotFoundError):
                logger.debug(f"Permission denied or file not found: {file_path}")
                results.append(("", "unknown", 0, str(file_path)))
            except Exception as e:
                logger.debug(f"Unexpected error processing {file_path}: {e}")
                results.append(("", "unknown", 0, str(file_path)))
        logger.debug(f"Completed batch processing. Got {len(results)} results")
        return results

    def print_status(self):
        """Print current status including I/O stats."""
        last_count = 0
        last_time = time.time()
        stall_threshold = 30  # seconds
        
        while True:
            if self.start_time is None:
                time.sleep(1)
                continue
                
            current_time = time.time()
            elapsed = current_time - self.start_time
            files_per_second = self.file_count / elapsed if elapsed > 0 else 0
            
            # Check for stalls
            if self.file_count == last_count:
                stall_time = current_time - last_time
                if stall_time > stall_threshold:
                    logger.warning(
                        f"Processing appears to be stalled! No new files processed in {stall_time:.1f} seconds."
                    )
            else:
                last_time = current_time
                last_count = self.file_count
            
            # Print simplified status without drive I/O stats
            logger.info(
                f"Processed {self.file_count} files ({self._format_size(self.total_size)}) | "
                f"Speed: {files_per_second:.1f} files/sec | "
                f"Elapsed time: {elapsed:.1f}s"
            )
            
            time.sleep(5)  # Update every 5 seconds

    def analyze_directory(self):
        """Analyze the directory structure and collect file information using optimized multiprocessing."""
        logger.info(f"Analyzing directory: {self.root_dir}")
        
        if self.load_cache():
            logger.info("Loaded analysis results from cache.")
            return

        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_interrupt)
        
        self.start_time = time.time()
        status_thread = threading.Thread(target=self.print_status, daemon=True)
        status_thread.start()

        # Collect all files to process
        files_to_process = []
        large_files = []
        
        logger.info("Scanning directory for files...")
        for root, dirs, files in os.walk(self.root_dir):
            if self.should_exit:
                break
            dirs[:] = [d for d in dirs if not self.should_ignore(Path(root) / d)]
            for file in files:
                file_path = Path(root) / file
                if not self.should_ignore(file_path) and str(file_path) not in self.processed_files:
                    try:
                        size = file_path.stat().st_size
                        if size > LARGE_FILE_THRESHOLD:
                            if not self.skip_large:
                                large_files.append(file_path)
                        else:
                            files_to_process.append(file_path)
                            
                        # In test mode, limit the number of files
                        if self.test_mode and len(files_to_process) + len(large_files) >= 2000:
                            logger.info("Test mode: Reached 2000 files limit")
                            break
                            
                    except (PermissionError, FileNotFoundError):
                        continue
                    except Exception as e:
                        logger.debug(f"Unexpected error accessing {file_path}: {e}")
            
            # Break out of directory walk if we've reached the test mode limit
            if self.test_mode and len(files_to_process) + len(large_files) >= 2000:
                break

        total_files = len(files_to_process) + len(large_files)
        logger.info(f"Found {total_files} files to analyze ({len(large_files)} large files)")

        # Process small files first
        if files_to_process:
            logger.info(f"Processing {len(files_to_process)} small files in batches of {BATCH_SIZE}...")
            
            # Process files in batches using ProcessPoolExecutor with sliding window
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Calculate optimal number of pending batches (2x workers for good throughput)
                max_pending_batches = MAX_WORKERS * 2
                
                # Create batches
                batches = [files_to_process[i:i + BATCH_SIZE] for i in range(0, len(files_to_process), BATCH_SIZE)]
                total_batches = len(batches)
                
                # Initialize tracking variables
                futures = {}  # Map futures to batch indices
                next_batch_idx = 0
                completed_batches = 0
                
                # Process results as they complete with progress bar
                with tqdm(total=len(files_to_process), desc="Analyzing small files") as pbar:
                    while completed_batches < total_batches and not self.should_exit:
                        # Submit new batches if we have capacity
                        while len(futures) < max_pending_batches and next_batch_idx < total_batches:
                            batch_idx = next_batch_idx
                            batch = batches[batch_idx]
                            future = executor.submit(self.process_files_batch, batch)
                            futures[future] = batch_idx
                            next_batch_idx += 1
                        
                        # Wait for any future to complete
                        if futures:
                            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                            
                            # Process completed futures
                            for future in done:
                                batch_idx = futures.pop(future)
                                completed_batches += 1
                                
                                try:
                                    results = future.result()
                                    for file_hash, file_type, size, file_path in results:
                                        if file_hash:
                                            self.file_hashes[file_hash].append(Path(file_path))
                                        self.file_types[file_type].append(Path(file_path))
                                        self.total_size += size
                                        self.file_count += 1
                                        self.processed_files.add(file_path)
                                        pbar.update(1)
                                except Exception as e:
                                    logger.error(f"Error processing batch {batch_idx}: {e}")
                        else:
                            # No futures to wait for, small sleep to prevent CPU spinning
                            time.sleep(0.01)

        # Then process large files if not skipped
        if not self.should_exit and large_files and not self.skip_large:
            logger.info(f"Now processing {len(large_files)} large files...")
            # Sort large files by size to process smaller ones first
            large_files.sort(key=lambda x: x.stat().st_size)
            
            # Process large files in parallel using the same sliding window approach
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Calculate optimal number of pending batches (2x workers for good throughput)
                max_pending_batches = MAX_WORKERS * 2
                
                # Create batches of large files (smaller batch size for large files)
                large_batch_size = max(1, BATCH_SIZE // 10)  # Smaller batches for large files
                batches = [large_files[i:i + large_batch_size] for i in range(0, len(large_files), large_batch_size)]
                total_batches = len(batches)
                
                # Initialize tracking variables
                futures = {}  # Map futures to batch indices
                next_batch_idx = 0
                completed_batches = 0
                
                # Process results as they complete with progress bar
                with tqdm(total=len(large_files), desc="Analyzing large files") as pbar:
                    while completed_batches < total_batches and not self.should_exit:
                        # Submit new batches if we have capacity
                        while len(futures) < max_pending_batches and next_batch_idx < total_batches:
                            batch_idx = next_batch_idx
                            batch = batches[batch_idx]
                            future = executor.submit(self.process_files_batch, batch)
                            futures[future] = batch_idx
                            next_batch_idx += 1
                        
                        # Wait for any future to complete
                        if futures:
                            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                            
                            # Process completed futures
                            for future in done:
                                batch_idx = futures.pop(future)
                                completed_batches += 1
                                
                                try:
                                    results = future.result()
                                    for file_hash, file_type, size, file_path in results:
                                        if file_hash:
                                            self.file_hashes[file_hash].append(Path(file_path))
                                        self.file_types[file_type].append(Path(file_path))
                                        self.total_size += size
                                        self.file_count += 1
                                        self.processed_files.add(file_path)
                                        pbar.update(1)
                                except Exception as e:
                                    logger.error(f"Error processing large file batch {batch_idx}: {e}")
                        else:
                            # No futures to wait for, small sleep to prevent CPU spinning
                            time.sleep(0.01)

        self.save_cache()

    def find_duplicates(self) -> Dict[str, List[Path]]:
        """Find duplicate files based on their hash."""
        return {h: paths for h, paths in self.file_hashes.items() if len(paths) > 1}

    def generate_report(self) -> dict:
        """Generate a comprehensive report of the analysis."""
        duplicates = self.find_duplicates()
        self.duplicate_count = sum(len(paths) - 1 for paths in duplicates.items())

        # Create detailed duplicate groups with size information
        duplicate_groups = {}
        total_recoverable_space = 0
        for hash_value, paths in duplicates.items():
            try:
                size = paths[0].stat().st_size  # All files in group have same size
                duplicate_count = len(paths) - 1  # Number of duplicate files (excluding original)
                recoverable_space = size * duplicate_count
                total_recoverable_space += recoverable_space
                
                duplicate_groups[hash_value] = {
                    'paths': [str(p) for p in paths],
                    'size': size,
                    'size_human': self._format_size(size),
                    'count': len(paths),
                    'type': self.get_file_type(paths[0]),
                    'recoverable_space': recoverable_space,
                    'recoverable_space_human': self._format_size(recoverable_space)
                }
            except (PermissionError, FileNotFoundError):
                continue

        # Sort duplicate groups by size (largest first)
        sorted_duplicates = sorted(duplicate_groups.items(), key=lambda x: x[1]['size'], reverse=True)

        report = {
            "analysis_date": datetime.now().isoformat(),
            "root_directory": str(self.root_dir),
            "total_size_bytes": self.total_size,
            "total_size_human": self._format_size(self.total_size),
            "total_files": self.file_count,
            "duplicate_files": self.duplicate_count,
            "total_recoverable_space_bytes": total_recoverable_space,
            "total_recoverable_space_human": self._format_size(total_recoverable_space),
            "file_types": {
                file_type: len(files) 
                for file_type, files in self.file_types.items()
            },
            "duplicate_groups": dict(sorted_duplicates)  # Now includes size and other metadata
        }
        return report

    def _format_size(self, size_bytes: int) -> str:
        """Convert bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"

    def analyze_duplicates(self, min_size: int = 0, output_file: str = None) -> Dict:
        """Analyze duplicate files and return detailed information.
        
        Args:
            min_size: Minimum file size in bytes to consider (default: 0)
            output_file: Optional file to write the analysis to
            
        Returns:
            Dictionary containing duplicate analysis
        """
        duplicates = {}
        total_duplicate_size = 0
        
        # Find duplicates
        for file_hash, file_paths in self.file_hashes.items():
            if len(file_paths) > 1:  # Only consider actual duplicates
                # Get size of first file (they should all be the same size)
                try:
                    size = file_paths[0].stat().st_size
                    if size >= min_size:
                        duplicates[file_hash] = {
                            'paths': [str(p) for p in file_paths],
                            'size': size,
                            'count': len(file_paths),
                            'type': self.get_file_type(file_paths[0])
                        }
                        total_duplicate_size += size * (len(file_paths) - 1)  # Size of duplicates only
                except (PermissionError, FileNotFoundError):
                    continue
        
        # Sort duplicates by size (largest first)
        sorted_duplicates = sorted(duplicates.items(), key=lambda x: x[1]['size'], reverse=True)
        
        analysis = {
            'total_duplicate_files': sum(d['count'] - 1 for d in duplicates.values()),
            'total_duplicate_size': total_duplicate_size,
            'total_duplicate_size_human': self._format_size(total_duplicate_size),
            'duplicate_groups': len(duplicates),
            'duplicates': dict(sorted_duplicates)  # Convert back to dict but maintain order
        }
        
        # Write to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"Duplicate analysis written to {output_file}")
        
        return analysis

def main():
    parser = argparse.ArgumentParser(description='Analyze backup directory structure and find duplicates')
    parser.add_argument('directory', help='Root directory to analyze')
    parser.add_argument('--dry-run', action='store_true', default=True,
                      help='Run in dry-run mode (default: True)')
    parser.add_argument('--output', help='Output JSON file for the report')
    parser.add_argument('--cache', help='Custom cache file path')
    parser.add_argument('--force', action='store_true',
                      help='Force reanalysis ignoring cache')
    parser.add_argument('--skip-large', action='store_true',
                      help='Skip processing of large files (>100MB)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    parser.add_argument('--analyze-duplicates', action='store_true',
                      help='Analyze and list duplicate files')
    parser.add_argument('--min-size', type=int, default=0,
                      help='Minimum file size in bytes to consider for duplicate analysis')
    parser.add_argument('--duplicates-output', help='Output file for duplicate analysis')
    parser.add_argument('--test', action='store_true',
                      help='Run in test mode (process only 2000 files)')
    
    args = parser.parse_args()

    # Set up logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Set up output directory
    output_dir = Path("/Users/jgribble/code/projects/backup_analyzer")
    output_dir.mkdir(exist_ok=True)
    
    # Ensure all output files go to the output directory
    if args.output:
        args.output = str(output_dir / Path(args.output).name)
    if args.duplicates_output:
        args.duplicates_output = str(output_dir / Path(args.duplicates_output).name)
    if args.cache:
        args.cache = str(output_dir / Path(args.cache).name)

    analyzer = BackupAnalyzer(args.directory, dry_run=args.dry_run, cache_file=args.cache, 
                            force=args.force, skip_large=args.skip_large, test_mode=args.test)
    
    start_time = time.time()
    analyzer.analyze_directory()
    report = analyzer.generate_report()
    end_time = time.time()

    # Print summary to console
    print("\n=== Analysis Summary ===")
    print(f"Total files analyzed: {report['total_files']}")
    print(f"Total size: {report['total_size_human']}")
    print(f"Duplicate files found: {report['duplicate_files']}")
    print(f"Analysis time: {end_time - start_time:.2f} seconds")
    print("\nFile types found:")
    for file_type, count in report['file_types'].items():
        print(f"- {file_type}: {count} files")

    # Save detailed report to JSON if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed report saved to: {args.output}")

    # Analyze duplicates if requested
    if args.analyze_duplicates:
        print("\n=== Duplicate Analysis ===")
        duplicates = analyzer.analyze_duplicates(min_size=args.min_size, output_file=args.duplicates_output)
        print(f"Total duplicate files: {duplicates['total_duplicate_files']}")
        print(f"Total duplicate size: {duplicates['total_duplicate_size_human']}")
        print(f"Number of duplicate groups: {duplicates['duplicate_groups']}")
        
        # Print all duplicate groups ordered by size (largest first)
        print("\nAll duplicate groups (ordered by size, largest first):")
        for i, (hash, info) in enumerate(duplicates['duplicates'].items(), 1):
            print(f"\n{i}. Size: {analyzer._format_size(info['size'])}")
            print(f"   Type: {info['type']}")
            print(f"   Count: {info['count']} files")
            print("   Paths:")
            for path in info['paths']:
                print(f"   - {path}")

if __name__ == "__main__":
    main() 
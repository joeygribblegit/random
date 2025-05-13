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
SIZE_CHUNK = 1024 * 1024  # 1MB for size-based comparison

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
        
        # Add size-based tracking for quick duplicate detection
        self.size_to_files = defaultdict(list)  # Maps file size to list of files
        
        # Track skipped files and their sizes
        self.skipped_files = defaultdict(int)  # Maps reason to total size
        self.skipped_file_count = defaultdict(int)  # Maps reason to count
        
        # Define patterns to ignore - reduced list
        self.ignored_patterns = {
            '.git',  # Keep this to avoid processing git metadata
            '__pycache__'  # Keep this to avoid processing Python cache
        }
        
        # Set up output directory
        self.output_dir = Path("/Users/jgribble/code/projects/backup_analyzer")
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up cache file in the output directory if not specified
        if cache_file:
            self.cache_file = Path(cache_file)
            logger.info(f"Using cache file: {self.cache_file}")
        else:
            # Use the output directory for cache
            self.cache_file = self.output_dir / "cache.pkl"
        
        

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
        """Calculate MD5 hash of a file using optimized reading strategy."""
        md5_hash = hashlib.md5()  # Using MD5 instead of SHA-256 for speed
        try:
            file_size = file_path.stat().st_size
            
            # For large files, use memory-mapped reading
            if file_size > LARGE_FILE_THRESHOLD:
                with open(file_path, 'rb') as f:
                    # Read first and last 1MB for quick comparison
                    f.seek(0)
                    first_chunk = f.read(SIZE_CHUNK)
                    f.seek(-SIZE_CHUNK, 2)
                    last_chunk = f.read(SIZE_CHUNK)
                    md5_hash.update(first_chunk)
                    md5_hash.update(last_chunk)
            else:
                # For smaller files, use chunked reading
                with open(file_path, "rb") as f:
                    while chunk := f.read(CHUNK_SIZE):
                        md5_hash.update(chunk)
                        
            return md5_hash.hexdigest()
        except (PermissionError, FileNotFoundError):
            return ""
        except Exception as e:
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
        
        # First pass: collect file sizes
        size_map = {}
        for file_path in file_paths:
            try:
                size = file_path.stat().st_size
                size_map[file_path] = size
            except (PermissionError, FileNotFoundError):
                continue
            except Exception as e:
                logger.debug(f"Unexpected error accessing {file_path}: {e}")
                continue
        
        # Group files by size for quick duplicate detection
        size_groups = defaultdict(list)
        for file_path, size in size_map.items():
            size_groups[size].append(file_path)
        
        # Process each size group
        for size, paths in size_groups.items():
            if len(paths) == 1:
                # Single file of this size
                file_path = paths[0]
                file_type = self.get_file_type(file_path)
                # Generate a unique hash for single files
                file_hash = self.calculate_file_hash(file_path)
                if not file_hash:  # If hash calculation failed, use a unique identifier
                    file_hash = f"unique_{file_path.name}_{size}"
                results.append((file_hash, file_type, size, str(file_path)))
            else:
                # Multiple files of same size, need to hash
                for file_path in paths:
                    try:
                        file_hash = self.calculate_file_hash(file_path)
                        file_type = self.get_file_type(file_path)
                        if not file_hash:  # If hash calculation failed, use a unique identifier
                            file_hash = f"error_{file_path.name}_{size}"
                        results.append((file_hash, file_type, size, str(file_path)))
                    except Exception as e:
                        logger.debug(f"Error processing {file_path}: {e}")
                        # Generate a unique hash for files that failed to hash
                        file_hash = f"error_{file_path.name}_{size}"
                        results.append((file_hash, "unknown", size, str(file_path)))
        
        logger.debug(f"Completed batch processing. Got {len(results)} results")
        return results

    def print_status(self):
        """Print current status including I/O stats."""
        last_count = 0
        last_time = time.time()
        
        while True:
            if self.start_time is None:
                time.sleep(1)
                continue
                
            current_time = time.time()
            elapsed = current_time - self.start_time
            files_per_second = self.file_count / elapsed if elapsed > 0 else 0
            
            # Print simplified status without drive I/O stats
            logger.info(
                f"Processed {self.file_count} files ({self._format_size(self.total_size)}) | "
                f"Speed: {files_per_second:.1f} files/sec | "
                f"Elapsed time: {elapsed:.1f}s"
            )
            
            time.sleep(5)  # Update every 5 seconds

    def get_file_size(self, file_path: Path) -> int:
        """Get the total size of a file including extended attributes and metadata."""
        try:
            stat = file_path.stat()
            size = stat.st_size
            
            # Get extended attributes size
            try:
                xattr_list = os.listxattr(str(file_path))
                for attr in xattr_list:
                    size += len(os.getxattr(str(file_path), attr))
            except (OSError, AttributeError):
                pass  # Some filesystems don't support xattrs
                
            # Get ACL size if available
            try:
                acl = os.getxattr(str(file_path), 'system.posix_acl_access')
                size += len(acl)
            except (OSError, AttributeError):
                pass  # ACLs not supported or not present
                
            return size
        except (PermissionError, FileNotFoundError) as e:
            raise e
        except Exception as e:
            logger.warning(f"Error getting size for {file_path}: {e}")
            return 0

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
        total_raw_size = 0  # Track total size before any filtering
        total_files_seen = 0  # Track total number of files seen
        
        # Track files by type for debugging
        files_by_type = defaultdict(int)
        size_by_type = defaultdict(int)
        
        # Track large files specifically
        large_files_by_type = defaultdict(int)
        large_files_size = 0
        large_files_count = 0
        
        logger.info("Scanning directory for files...")
        try:
            for root, dirs, files in os.walk(self.root_dir, topdown=True):
                if self.should_exit:
                    break
                
                # Filter directories before processing
                dirs[:] = [d for d in dirs if not self.should_ignore(Path(root) / d)]
                
                for file in files:
                    total_files_seen += 1
                    file_path = Path(root) / file
                    try:
                        # Get file size including metadata
                        size = self.get_file_size(file_path)
                        total_raw_size += size
                        
                        # Track file type for debugging
                        file_type = self.get_file_type(file_path)
                        files_by_type[file_type] += 1
                        size_by_type[file_type] += size
                        
                        if self.should_ignore(file_path):
                            self.skipped_files['ignored_pattern'] += size
                            self.skipped_file_count['ignored_pattern'] += 1
                            continue
                            
                        if str(file_path) in self.processed_files:
                            self.skipped_files['already_processed'] += size
                            self.skipped_file_count['already_processed'] += 1
                            continue
                            
                        if size > LARGE_FILE_THRESHOLD:
                            if not self.skip_large:
                                large_files.append(file_path)
                                large_files_count += 1
                                large_files_size += size
                                large_files_by_type[file_type] += 1
                            else:
                                self.skipped_files['large_file'] += size
                                self.skipped_file_count['large_file'] += 1
                        else:
                            files_to_process.append(file_path)
                            
                        # In test mode, limit the number of files
                        if self.test_mode and len(files_to_process) + len(large_files) >= 2000:
                            logger.info("Test mode: Reached 2000 files limit")
                            break
                            
                    except PermissionError:
                        self.skipped_files['permission_error'] += size
                        self.skipped_file_count['permission_error'] += 1
                        logger.warning(f"Permission error accessing {file_path}")
                        continue
                    except FileNotFoundError:
                        self.skipped_files['not_found'] += size
                        self.skipped_file_count['not_found'] += 1
                        logger.warning(f"File not found: {file_path}")
                        continue
                    except Exception as e:
                        self.skipped_files['other_error'] += size
                        self.skipped_file_count['other_error'] += 1
                        logger.warning(f"Unexpected error accessing {file_path}: {e}")
                        continue
                
                # Break out of directory walk if we've reached the test mode limit
                if self.test_mode and len(files_to_process) + len(large_files) >= 2000:
                    break
                    
        except Exception as e:
            logger.error(f"Error during directory walk: {e}")
            raise

        total_files = len(files_to_process) + len(large_files)
        logger.info(f"\nDirectory scan complete:")
        logger.info(f"Total files seen: {total_files_seen}")
        logger.info(f"Total raw size: {self._format_size(total_raw_size)} ({total_raw_size} bytes)")
        logger.info(f"Files to process: {total_files} ({len(large_files)} large files)")
        
        # Log large files summary
        if large_files_count > 0:
            logger.info("\nLarge files summary:")
            logger.info(f"Total large files: {large_files_count}")
            logger.info(f"Total large files size: {self._format_size(large_files_size)} ({large_files_size} bytes)")
            logger.info("Large files by type:")
            for file_type, count in sorted(large_files_by_type.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {file_type}: {count} files")
        
        # Log file type distribution
        logger.info("\nFile type distribution:")
        for file_type, count in sorted(files_by_type.items(), key=lambda x: x[1], reverse=True):
            size = size_by_type[file_type]
            logger.info(f"  {file_type}: {count} files, {self._format_size(size)} ({size} bytes)")
        
        # Log skipped files summary
        if any(self.skipped_files.values()):
            logger.info("\nSkipped files summary:")
            total_skipped_size = 0
            for reason, size in self.skipped_files.items():
                count = self.skipped_file_count[reason]
                total_skipped_size += size
                logger.info(f"  {reason}: {count} files, {self._format_size(size)} ({size} bytes)")
            logger.info(f"Total skipped size: {self._format_size(total_skipped_size)} ({total_skipped_size} bytes)")
        
        # Process all files in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Combine all files into one list
            all_files = files_to_process + large_files
            
            # Create batches
            batches = [all_files[i:i + BATCH_SIZE] for i in range(0, len(all_files), BATCH_SIZE)]
            total_batches = len(batches)
            
            # Track failed batches for retry
            failed_batches = []
            max_retries = 3
            
            # Process results as they complete with progress bar
            with tqdm(total=len(all_files), desc="Analyzing files") as pbar:
                # Submit all batches first
                futures = []
                for batch in batches:
                    future = executor.submit(self.process_files_batch, batch)
                    future.batch = batch  # Store the batch with the future
                    futures.append(future)
                
                # Process results as they complete
                for future in as_completed(futures):
                    if self.should_exit:
                        break
                        
                    try:
                        results = future.result()
                        # Create a temporary dictionary to store new entries
                        new_entries = defaultdict(list)
                        new_types = defaultdict(list)
                        
                        for file_hash, file_type, size, file_path in results:
                            if file_hash:
                                new_entries[file_hash].append(Path(file_path))
                            new_types[file_type].append(Path(file_path))
                            self.total_size += size
                            self.file_count += 1
                            self.processed_files.add(file_path)
                            pbar.update(1)
                        
                        # Update the main dictionaries after processing the batch
                        for hash_value, paths in new_entries.items():
                            self.file_hashes[hash_value].extend(paths)
                        for file_type, paths in new_types.items():
                            self.file_types[file_type].extend(paths)
                            
                    except Exception as e:
                        logger.error(f"Error processing batch: {e}")
                        # Add to failed batches for retry
                        failed_batches.append(future)
                
                # Retry failed batches
                retry_count = 0
                while failed_batches and retry_count < max_retries:
                    retry_count += 1
                    logger.info(f"Retrying {len(failed_batches)} failed batches (attempt {retry_count}/{max_retries})")
                    
                    # Create new futures for failed batches
                    new_futures = []
                    for future in failed_batches:
                        try:
                            batch = future.batch  # Get the original batch
                            new_future = executor.submit(self.process_files_batch, batch)
                            new_future.batch = batch  # Store the batch with the new future
                            new_futures.append(new_future)
                        except Exception as e:
                            logger.error(f"Error creating retry future: {e}")
                    
                    # Clear failed batches list
                    failed_batches = []
                    
                    # Process retry results
                    for future in as_completed(new_futures):
                        try:
                            results = future.result()
                            # Create a temporary dictionary to store new entries
                            new_entries = defaultdict(list)
                            new_types = defaultdict(list)
                            
                            for file_hash, file_type, size, file_path in results:
                                if file_hash:
                                    new_entries[file_hash].append(Path(file_path))
                                new_types[file_type].append(Path(file_path))
                                self.total_size += size
                                self.file_count += 1
                                self.processed_files.add(file_path)
                                pbar.update(1)
                            
                            # Update the main dictionaries after processing the batch
                            for hash_value, paths in new_entries.items():
                                self.file_hashes[hash_value].extend(paths)
                            for file_type, paths in new_types.items():
                                self.file_types[file_type].extend(paths)
                                
                        except Exception as e:
                            logger.error(f"Error processing retry batch: {e}")
                            failed_batches.append(future)
                
                # Log any remaining failed batches
                if failed_batches:
                    logger.warning(f"Failed to process {len(failed_batches)} batches after {max_retries} retries")

        # Final size summary
        logger.info("\nFinal size summary:")
        logger.info(f"Total raw size: {self._format_size(total_raw_size)} ({total_raw_size} bytes)")
        logger.info(f"Total skipped size: {self._format_size(total_skipped_size)} ({total_skipped_size} bytes)")
        logger.info(f"Total processed size: {self._format_size(self.total_size)} ({self.total_size} bytes)")
        logger.info(f"Size difference: {self._format_size(total_raw_size - self.total_size - total_skipped_size)} ({total_raw_size - self.total_size - total_skipped_size} bytes)")

        self.save_cache()

    def find_duplicates(self) -> Dict[str, List[Path]]:
        """Find duplicate files based on their hash."""
        return {h: paths for h, paths in self.file_hashes.items() if len(paths) > 1}

    def generate_report(self) -> dict:
        """Generate a comprehensive report of the analysis."""
        duplicates = self.find_duplicates()
        self.duplicate_count = sum(len(paths) - 1 for paths in duplicates.items())
        
        # Calculate additional duplicate statistics
        total_files_with_duplicates = sum(1 for paths in duplicates.values() if len(paths) > 1)
        duplicate_distribution = defaultdict(int)
        for paths in duplicates.values():
            if len(paths) > 1:
                duplicate_distribution[len(paths)] += 1

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
            "files_with_duplicates": total_files_with_duplicates,
            "duplicate_distribution": dict(duplicate_distribution),
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
        """Convert bytes to human readable format using binary prefixes (KiB, MiB, GiB)."""
        for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PiB"

    def analyze_duplicate_folders(self, duplicates: Dict[str, List[Path]]) -> Dict:
        """Analyze duplicate folders based on duplicate files.
        
        Args:
            duplicates: Dictionary of duplicate file groups
            
        Returns:
            Dictionary containing duplicate folder analysis
        """
        # Group duplicate files by their parent folders
        folder_groups = defaultdict(list)
        for hash_value, paths in duplicates.items():
            # Get parent folders for each path
            parent_folders = [str(p.parent) for p in paths]
            # Add to folder groups if we have multiple unique parent folders
            if len(set(parent_folders)) > 1:
                # Use a string key instead of tuple
                folder_key = "|".join(sorted(set(parent_folders)))
                folder_groups[folder_key].append({
                    'hash': hash_value,
                    'paths': [str(p) for p in paths],
                    'size': paths[0].stat().st_size,
                    'type': self.get_file_type(paths[0])
                })
        
        # Analyze folder relationships
        folder_analysis = {}
        for folder_key, files in folder_groups.items():
            folders = folder_key.split("|")
            # Calculate folder similarity scores
            folder_sets = [set(f.split('/')) for f in folders]
            similarity_scores = {}
            
            # Compare each pair of folders
            for i, folder1 in enumerate(folders):
                for j, folder2 in enumerate(folders[i+1:], i+1):
                    set1 = set(folder1.split('/'))
                    set2 = set(folder2.split('/'))
                    
                    # Calculate Jaccard similarity
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    similarity = intersection / union if union > 0 else 0
                    
                    # Check if one folder is a subset of another
                    is_subset = set1.issubset(set2) or set2.issubset(set1)
                    
                    similarity_scores[f"{folder1} <-> {folder2}"] = {
                        'similarity': similarity,
                        'is_subset': is_subset,
                        'subset_direction': 'folder1' if set1.issubset(set2) else 'folder2' if set2.issubset(set1) else None
                    }
            
            # Calculate total size of duplicate files in these folders
            total_size = sum(f['size'] * (len(f['paths']) - 1) for f in files)
            
            folder_analysis[folder_key] = {
                'folders': folders,  # Store the list of folders
                'files': files,
                'total_files': len(files),
                'total_size': total_size,
                'total_size_human': self._format_size(total_size),
                'similarity_scores': similarity_scores
            }
        
        return folder_analysis

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
                        duplicates[file_hash] = file_paths
                        total_duplicate_size += size * (len(file_paths) - 1)  # Size of duplicates only
                except (PermissionError, FileNotFoundError):
                    continue
        
        # Analyze duplicate folders
        folder_analysis = self.analyze_duplicate_folders(duplicates)
        
        # Convert duplicates to the format expected by the output
        formatted_duplicates = {}
        for hash_value, paths in duplicates.items():
            size = paths[0].stat().st_size
            formatted_duplicates[hash_value] = {
                'paths': [str(p) for p in paths],
                'size_bytes': size,
                'size_human': self._format_size(size),
                'count': len(paths),
                'type': self.get_file_type(paths[0]),
                'total_duplicate_size_bytes': size * (len(paths) - 1),
                'total_duplicate_size_human': self._format_size(size * (len(paths) - 1))
            }
        
        # Sort duplicates by size (largest first)
        sorted_duplicates = sorted(formatted_duplicates.items(), 
                                 key=lambda x: x[1]['size_bytes'], 
                                 reverse=True)
        
        analysis = {
            'total_duplicate_files': sum(d['count'] - 1 for d in formatted_duplicates.values()),
            'total_duplicate_size_bytes': total_duplicate_size,
            'total_duplicate_size_human': self._format_size(total_duplicate_size),
            'duplicate_groups': len(duplicates),
            'duplicates': dict(sorted_duplicates),  # Convert back to dict but maintain order
            'folder_analysis': folder_analysis
        }
        
        # Write to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"Duplicate analysis written to {output_file}")
        
        return analysis

    def categorize_file(self, file_path: Path) -> str:
        """Categorize a file into a high-level category."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type:
            return "other"
            
        main_type = mime_type.split('/')[0]
        if main_type in ['image', 'video', 'audio']:
            return 'media'
        elif main_type in ['text', 'application']:
            # Further categorize application files
            if mime_type.endswith(('pdf', 'msword', 'vnd.openxmlformats-officedocument')):
                return 'documents'
            elif mime_type.endswith(('octet-stream', 'x-msdownload', 'x-msdos-program')):
                return 'applications'
            return 'documents'
        return 'other'

    def analyze_other_files(self, output_file: str = None) -> dict:
        """Analyze and categorize files that are not media or documents.
        
        Args:
            output_file: Optional file to write the analysis to
            
        Returns:
            Dictionary containing analysis of other files
        """
        # Initialize categories
        categories = {
            'media': [],
            'documents': [],
            'applications': [],
            'other': []
        }
        
        # Track file types and extensions within each category
        file_types_by_category = {
            'media': defaultdict(int),
            'documents': defaultdict(int),
            'applications': defaultdict(int),
            'other': defaultdict(int)
        }
        
        # Track file extensions for unknown types
        extensions_by_category = {
            'media': defaultdict(int),
            'documents': defaultdict(int),
            'applications': defaultdict(int),
            'other': defaultdict(int)
        }
        
        # Process all files
        for file_paths in self.file_hashes.values():
            for file_path in file_paths:
                try:
                    size = file_path.stat().st_size
                    category = self.categorize_file(file_path)
                    file_type = self.get_file_type(file_path)
                    extension = file_path.suffix.lower() if file_path.suffix else 'no_extension'
                    
                    # Track file type counts
                    file_types_by_category[category][file_type] += 1
                    
                    # Track extensions for unknown types
                    if file_type == 'unknown':
                        extensions_by_category[category][extension] += 1
                    
                    categories[category].append({
                        'path': str(file_path),
                        'size': size,
                        'size_human': self._format_size(size),
                        'type': file_type,
                        'extension': extension
                    })
                except (PermissionError, FileNotFoundError):
                    continue
        
        # Sort each category by size (largest first)
        for category in categories:
            categories[category].sort(key=lambda x: x['size'], reverse=True)
        
        # Calculate totals and include file type and extension breakdowns
        totals = {
            category: {
                'count': len(files),
                'total_size': sum(f['size'] for f in files),
                'total_size_human': self._format_size(sum(f['size'] for f in files)),
                'file_types': dict(file_types_by_category[category]),
                'extensions': dict(extensions_by_category[category])
            }
            for category, files in categories.items()
        }
        
        analysis = {
            'totals': totals,
            'categories': categories
        }
        
        # Write to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"Other files analysis written to {output_file}")
        
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
    parser.add_argument('--analyze-other', action='store_true',
                      help='Analyze and categorize other files')
    parser.add_argument('--other-output', help='Output file for other files analysis')
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
    if args.other_output:
        args.other_output = str(output_dir / Path(args.other_output).name)
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
        
        # Display folder analysis
        print("\n=== Duplicate Folder Analysis ===")
        folder_analysis = duplicates['folder_analysis']
        print(f"Found {len(folder_analysis)} groups of duplicate folders")
        
        # Sort folder groups by total size
        sorted_folders = sorted(folder_analysis.items(), 
                              key=lambda x: x[1]['total_size'], 
                              reverse=True)
        
        for folder_key, analysis in sorted_folders[:10]:  # Show top 10 largest groups
            print(f"\nFolder Group (Total Size: {analysis['total_size_human']}):")
            for folder in analysis['folders']:
                print(f"  - {folder}")
            
            print("\n  Similarity Analysis:")
            for pair, scores in analysis['similarity_scores'].items():
                print(f"    {pair}:")
                print(f"      Similarity: {scores['similarity']:.2%}")
                if scores['is_subset']:
                    print(f"      One folder is a subset of the other")
                    print(f"      Subset direction: {scores['subset_direction']}")
            
            print(f"  Total duplicate files in group: {analysis['total_files']}")

    # Analyze other files if requested
    if args.analyze_other:
        print("\n=== Other Files Analysis ===")
        other_files = analyzer.analyze_other_files(output_file=args.other_output)
        print("\nCategory Totals:")
        for category, stats in other_files['totals'].items():
            print(f"{category.title()}:")
            print(f"  Count: {stats['count']}")
            print(f"  Total Size: {stats['total_size_human']}")
        
        print("\nLargest Files by Category:")
        for category, files in other_files['categories'].items():
            if files:
                print(f"\n{category.title()} (Top 5):")
                for i, file_info in enumerate(files[:5], 1):
                    print(f"  {i}. {file_info['size_human']} - {file_info['path']}")

if __name__ == "__main__":
    main() 
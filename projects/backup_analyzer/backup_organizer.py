#!/usr/bin/env python3

import os
import json
import shutil
from pathlib import Path
import logging
from datetime import datetime
import mimetypes
from typing import Dict, List, Set
import argparse
import pickle
from collections import defaultdict
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BackupOrganizer:
    def __init__(self, source_dir: str, target_dir: str, dry_run: bool = True, cache_file: str = None, limit: int = None):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.dry_run = dry_run
        self.copies_log = []
        self.start_time = datetime.now()
        self.limit = limit
        
        # Track used folder names for quick lookup
        self.used_picture_folders = {}  # Maps source path to target folder name
        self.used_document_folders = {}  # Maps source path to target folder name
        
        # Set up cache file
        if cache_file:
            self.cache_file = Path(cache_file)
        else:
            # Use default cache location from analyzer
            self.cache_file = Path("/Users/jgribble/code/projects/backup_analyzer/cache.pkl")
        
        # Create target directories
        self.pictures_dir = self.target_dir / "Pictures"
        self.videos_dir = self.target_dir / "Videos"
        self.other_dir = self.target_dir / "Other"
        
        if not self.dry_run:
            try:
                logger.info(f"Creating target directories...")
                self.pictures_dir.mkdir(parents=True, exist_ok=True)
                self.videos_dir.mkdir(parents=True, exist_ok=True)
                self.other_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Target directories created successfully")
            except Exception as e:
                logger.error(f"Failed to create target directories: {e}")
                raise
        
        # Load cached data
        self.file_hashes = defaultdict(list)
        self.load_cache()
        
        # Track used filenames and their hashes for collision handling
        self.used_filenames = {}  # Maps filename to its hash
        
        # Create optimized hash maps for quick lookup
        self.source_to_hash = {}  # Maps source path to its hash
        self.hash_to_first_copy = {}  # Maps hash to the first copy's target path
        self.size_to_files = defaultdict(list)  # Maps file size to list of files
        
        # Initialize the hash maps
        for hash_value, paths in self.file_hashes.items():
            for path in paths:
                path_str = str(path)
                self.source_to_hash[path_str] = hash_value
                try:
                    size = Path(path).stat().st_size
                    self.size_to_files[size].append(path)
                except (PermissionError, FileNotFoundError):
                    continue
        
        # Create log directory and initialize log file
        self.log_dir = Path("/Users/jgribble/code/projects/backup_analyzer/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"copies_log_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        
        # Load previous copies and track already copied files
        self.already_copied_files = set()
        self._load_previous_copies()
        
        # --- Custom Rules Configuration ---
        self.skip_patterns = [
            ".class", "/dos/", ".prt.", ".prt", "Thumbs.db", "/gossip queer/",
            ".DS_Store",
            "/Hailey Camp/",
            "/HAILEY8TH GRADE./",
            "/Hailey/",
            "/Joey Movies Backup/driving/",
            "/tesla_video/",
            "/wedding videos/",
            "/ICP Fall 2006/",
            "/SCIS GRAD PHOTOS/",
            "/Camp Casey/",
            "/dad/",
            "/KJH 2007-08/",
            "/KJH 2008-09/",
            "/MUGS2009-10/",
            "/Boat Bash 09/",
            "Boat Bash/",
        ]
        self.move_rules = [
            {'pattern': "/From School/", 'destination_folder_name': "From School"},
            # Add other move rules here, for example:
            # {'pattern': "MUGS2009", 'destination_folder_name': "Middle School/yearbook"},
            # {'pattern': "Mac Backup/Hailey/", 'destination_folder_name': "Hailey Backup"},
        ]
        # --- End Custom Rules Configuration ---

        # Initialize counters for custom rules
        self.skipped_by_pattern_count = 0
        self.skipped_by_pattern_size = 0
        self.moved_by_pattern_count = 0
        self.moved_by_pattern_size = 0
        
        self._save_log()  # Initialize empty log file
    
    def load_cache(self) -> bool:
        """Load cached analysis results if they exist."""
        if not os.path.exists(self.cache_file):
            logger.error(f"Cache file not found: {self.cache_file}")
            return False
            
        try:
            with open(self.cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                
            self.file_hashes = cached_data['file_hashes']
            logger.info(f"Loaded {len(self.file_hashes)} file groups from cache")
            return True
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return False
    
    def is_media_file(self, file_path: Path) -> bool:
        """Check if a file is a media file."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type:
            return False
        return mime_type.startswith(('image/', 'video/', 'audio/'))
    
    def is_document_file(self, file_path: Path) -> bool:
        """Check if a file is a document file."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type:
            return False
        return mime_type.startswith(('text/', 'application/pdf', 'application/msword', 
                                   'application/vnd.openxmlformats-officedocument'))
    
    def is_picture_file(self, file_path: Path) -> bool:
        """Check if a file is a picture file."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type:
            return False
        return mime_type.startswith('image/')
    
    def is_video_file(self, file_path: Path) -> bool:
        """Check if a file is a video file."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type:
            return False
        return mime_type.startswith('video/')
    
    def get_relative_path(self, file_path: Path) -> Path:
        """Get the path relative to the source directory."""
        return file_path.relative_to(self.source_dir)
    
    def should_process_file(self, file_path: Path) -> bool:
        """Check if a file should be processed."""
        return self.is_media_file(file_path) or self.is_document_file(file_path)
    
    def get_unique_filename(self, target_dir: Path, filename: str, file_hash: str) -> str:
        """Get a unique filename by checking hash before incrementing."""
        base_name = filename
        counter = 1
        
        # Check if we've seen this filename before
        if filename in self.used_filenames:
            # If the hash matches, use the same filename
            if self.used_filenames[filename] == file_hash:
                return filename
            # If hash is different, we need to increment
            while True:
                name, ext = os.path.splitext(base_name)
                new_filename = f"{name}_{counter}{ext}"
                # Check if this new filename is already used
                if new_filename not in self.used_filenames:
                    self.used_filenames[new_filename] = file_hash
                    return new_filename
                # If the new filename exists but has the same hash, use it
                if self.used_filenames[new_filename] == file_hash:
                    return new_filename
                # Otherwise, increment and try again
                counter += 1
        else:
            # First time seeing this filename
            self.used_filenames[filename] = file_hash
            return filename
    
    def get_target_path(self, file_path: Path, custom_move_folder_name: str = None) -> Path:
        """Get the target path for a file, handling filename collisions based on hash."""
        # Determine target directory based on file type or custom move rule
        if custom_move_folder_name:
            target_dir = self.target_dir / custom_move_folder_name
            logger.debug(f"Using custom target directory for {file_path}: {target_dir}")
        elif self.is_picture_file(file_path):
            target_dir = self.pictures_dir
        elif self.is_video_file(file_path):
            target_dir = self.videos_dir
        else:
            target_dir = self.other_dir
        
        # Get the file's hash
        file_hash = self.source_to_hash.get(str(file_path))
        if file_hash is None:
            self.skipped_no_hash += 1
            logger.warning(f"No hash found for file: {file_path}")
            return None
        
        # If we've already copied a file with this hash, skip it
        if file_hash in self.hash_to_first_copy:
            self.skipped_by_hash += 1
            logger.debug(f"Skipping duplicate file (same hash): {file_path}")
            return None
        
        # Get the filename and make it unique based on hash
        filename = self.get_unique_filename(target_dir, file_path.name, file_hash)
        return target_dir / filename
    
    def copy_file(self, source_path: Path, target_path: Path) -> bool:
        """Copy a file to its target location."""
        try:
            # Check if source file exists
            if not source_path.exists():
                logger.warning(f"Source file does not exist: {source_path}")
                return False
                
            # Check if source file is readable
            if not os.access(source_path, os.R_OK):
                logger.warning(f"Source file is not readable: {source_path}")
                return False

            # Create parent directories if they don't exist
            if not self.dry_run:
                try:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logger.error(f"Failed to create directory {target_path.parent}: {e}")
                    return False
            
            # Copy the file using optimized method
            if not self.dry_run:
                # Use sendfile for faster copying on Unix systems
                if os.name == 'posix':
                    try:
                        with open(source_path, 'rb') as src, open(target_path, 'wb') as dst:
                            # Get file size
                            src_size = os.fstat(src.fileno()).st_size
                            # Use sendfile for faster copying
                            os.sendfile(dst.fileno(), src.fileno(), 0, src_size)
                    except (OSError, AttributeError):
                        # Fallback to shutil.copy2 if sendfile fails
                        shutil.copy2(str(source_path), str(target_path))
                else:
                    # Use shutil.copy2 for non-Unix systems
                    shutil.copy2(str(source_path), str(target_path))
            
            # Update hash_to_first_copy with this copy
            source_str = str(source_path)
            if source_str in self.source_to_hash:
                hash_value = self.source_to_hash[source_str]
                self.hash_to_first_copy[hash_value] = str(target_path)
            
            # Log the copy (get size once and reuse)
            size = source_path.stat().st_size
            self.copies_log.append({
                'source': str(source_path),
                'target': str(target_path),
                'size_bytes': size,
                'size_human': self._format_size(size),
                'timestamp': datetime.now().isoformat()
            })
            
            return True
        except Exception as e:
            logger.error(f"Error copying {source_path}: {e}")
            return False
    
    def _load_previous_copies(self):
        """Load previous copies from log files to track already copied files."""
        try:
            # Get all log files in the log directory
            log_files = sorted(self.log_dir.glob("copies_log_*.json"), reverse=True)
            
            for log_file in log_files:
                try:
                    with open(log_file, 'r') as f:
                        log_data = json.load(f)
                        
                    # Add all previously copied files to our set
                    for copy in log_data.get('copies', []):
                        self.already_copied_files.add(copy['source'])
                        
                        # Update hash_to_first_copy map with existing copies
                        source_path = copy['source']
                        if source_path in self.source_to_hash:
                            hash_value = self.source_to_hash[source_path]
                            if hash_value not in self.hash_to_first_copy:
                                self.hash_to_first_copy[hash_value] = copy['target']
                        
                        # Ensure all entries have both size formats
                        if 'size' in copy:
                            size_bytes = copy['size']
                            copy['size_bytes'] = size_bytes
                            copy['size_human'] = self._format_size(size_bytes)
                            del copy['size']
                        
                    # Also load the copies into our current log
                    self.copies_log.extend(log_data.get('copies', []))
                    
                    logger.info(f"Loaded {len(log_data.get('copies', []))} copies from {log_file}")
                except Exception as e:
                    logger.error(f"Error loading log file {log_file}: {e}")
                    continue
                    
            logger.info(f"Total of {len(self.already_copied_files)} files already copied")
        except Exception as e:
            logger.error(f"Error loading previous copies: {e}")

    def organize_files(self):
        """Organize files into media and document categories."""
        logger.info(f"Starting organization from {self.source_dir} to {self.target_dir}")
        if self.dry_run:
            logger.info("Running in dry-run mode - no files will be copied")
        if self.limit:
            logger.info(f"Processing only {self.limit} files")
        
        # Check if source directory exists
        if not self.source_dir.exists():
            logger.error(f"Source directory does not exist: {self.source_dir}")
            return
            
        # Check if source directory is readable
        if not os.access(self.source_dir, os.R_OK):
            logger.error(f"Source directory is not readable: {self.source_dir}")
            return
        
        # Create target directories if not in dry run mode
        if not self.dry_run:
            try:
                logger.info(f"Creating target directories...")
                self.pictures_dir.mkdir(parents=True, exist_ok=True)
                self.videos_dir.mkdir(parents=True, exist_ok=True)
                self.other_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Target directories created successfully")
            except Exception as e:
                logger.error(f"Failed to create target directories: {e}")
                return
        
        # Initialize counters
        self.total_files = 0
        self.copied_files = 0
        self.total_size = 0
        self.skipped_files = 0
        self.skipped_by_type = 0
        self.skipped_by_hash = 0
        self.skipped_no_hash = 0
        self.skipped_already_copied = 0
        self.skipped_not_exists = 0
        self.skipped_not_media = 0
        
        # Track sizes by type
        self.picture_size = 0
        self.video_size = 0
        self.other_size = 0
        self.picture_count = 0
        self.video_count = 0
        self.other_count = 0
        
        # Track skipped sizes
        self.skipped_size = 0
        self.skipped_by_hash_size = 0
        self.skipped_no_hash_size = 0
        self.skipped_already_copied_size = 0
        self.skipped_not_exists_size = 0
        
        # Track total source size and files not in cache
        self.total_source_size = 0
        self.total_source_files = 0
        self.files_not_in_cache = 0
        self.size_not_in_cache = 0
        self.files_not_in_cache_by_type = defaultdict(int)
        self.size_not_in_cache_by_type = defaultdict(int)
        
        # First, collect all valid files to process
        files_to_process = []
        total_files_seen = 0
        
        # Calculate total source size and track files not in cache
        logger.info("Calculating total source size and checking cache coverage...")
        for root, dirs, files in os.walk(self.source_dir):
            for file in files:
                try:
                    file_path = Path(root) / file
                    size = file_path.stat().st_size
                    self.total_source_size += size
                    self.total_source_files += 1
                    
                    # Check if file is in cache
                    if str(file_path) not in self.source_to_hash:
                        self.files_not_in_cache += 1
                        self.size_not_in_cache += size
                        
                        # Track by type
                        if self.is_picture_file(file_path):
                            self.files_not_in_cache_by_type['pictures'] += 1
                            self.size_not_in_cache_by_type['pictures'] += size
                        elif self.is_video_file(file_path):
                            self.files_not_in_cache_by_type['videos'] += 1
                            self.size_not_in_cache_by_type['videos'] += size
                        else:
                            self.files_not_in_cache_by_type['other'] += 1
                            self.size_not_in_cache_by_type['other'] += size
                            
                except (PermissionError, FileNotFoundError):
                    continue
        
        logger.info(f"Total source size: {self._format_size(self.total_source_size)} ({self.total_source_files} files)")
        logger.info(f"Files not in cache: {self.files_not_in_cache} files, {self._format_size(self.size_not_in_cache)}")
        logger.info("Files not in cache by type:")
        for file_type, count in self.files_not_in_cache_by_type.items():
            size = self.size_not_in_cache_by_type[file_type]
            logger.info(f"  {file_type}: {count} files, {self._format_size(size)}")
        
        # Process files directly from file_hashes
        for file_paths in self.file_hashes.values():
            for file_path in file_paths:
                total_files_seen += 1
                file_path = Path(file_path)
                
                try:
                    size = file_path.stat().st_size
                    
                    # Skip if file has already been copied
                    if str(file_path) in self.already_copied_files:
                        self.skipped_already_copied += 1
                        self.skipped_already_copied_size += size
                        logger.debug(f"Skipping already copied file: {file_path}")
                        continue
                    
                    # Skip if file doesn't exist
                    if not file_path.exists():
                        self.skipped_not_exists += 1
                        self.skipped_not_exists_size += size
                        logger.debug(f"Skipping non-existent file: {file_path}")
                        continue
                        
                    files_to_process.append(file_path)
                    logger.debug(f"Added file to process: {file_path}")
                except (PermissionError, FileNotFoundError):
                    continue
        
        logger.info(f"\nFile Processing Summary:")
        logger.info(f"Total source files: {self.total_source_files}")
        logger.info(f"Total source size: {self._format_size(self.total_source_size)}")
        logger.info(f"Files in cache: {total_files_seen}")
        logger.info(f"Files to process: {len(files_to_process)}")
        logger.info(f"Skipped (already copied): {self.skipped_already_copied} files, {self._format_size(self.skipped_already_copied_size)}")
        logger.info(f"Skipped (not exists): {self.skipped_not_exists} files, {self._format_size(self.skipped_not_exists_size)}")
        
        # Process files with progress bar
        limit_reached_flag = False # Flag to break outer loop if limit is hit
        with tqdm(total=len(files_to_process), desc="Processing files") as pbar:
            for file_path in files_to_process:
                self.total_files += 1
                current_file_size = 0 # Initialize for potential skip before stat
                try:
                    current_file_size = file_path.stat().st_size
                except (FileNotFoundError, PermissionError) as e:
                    logger.warning(f"Could not get size for {file_path} before processing: {e}")
                    # Decide if we should skip or try to process without size first
                    # For now, let's log and attempt to proceed if possible, or it will fail later

                # --- Apply Custom Skip Rules ---
                is_skipped_by_rule = False
                for pattern in self.skip_patterns:
                    if pattern in str(file_path):
                        logger.info(f"Skipping file '{file_path}' due to skip pattern: '{pattern}'")
                        self.skipped_by_pattern_count += 1
                        if current_file_size > 0 : # Only add size if we got it
                             self.skipped_by_pattern_size += current_file_size
                        is_skipped_by_rule = True
                        break
                if is_skipped_by_rule:
                    pbar.update(1)
                    continue
                # --- End Custom Skip Rules ---

                # --- Apply Custom Move Rules ---
                custom_move_folder = None
                for rule in self.move_rules:
                    if rule['pattern'] in str(file_path):
                        custom_move_folder = rule['destination_folder_name']
                        logger.info(f"File '{file_path}' matches move rule '{rule['pattern']}', target folder: '{custom_move_folder}'")
                        break
                # --- End Custom Move Rules ---
                
                target_path = self.get_target_path(file_path, custom_move_folder_name=custom_move_folder)
                
                if target_path is None:
                    # File was skipped by get_target_path logic (duplicate hash, no hash)
                    self.skipped_files += 1 # Overall skipped files count
                    if current_file_size > 0:
                        self.skipped_size += current_file_size # Overall skipped size

                    # Attribute size to specific skip reason based on get_target_path's logic
                    # The counts (self.skipped_by_hash, self.skipped_no_hash) are done in get_target_path
                    _file_hash_temp_check = self.source_to_hash.get(str(file_path))
                    if _file_hash_temp_check is None:
                        # This implies it was skipped by get_target_path due to no hash
                        if current_file_size > 0:
                            self.skipped_no_hash_size += current_file_size
                    elif _file_hash_temp_check in self.hash_to_first_copy:
                        # This implies it was skipped by get_target_path due to duplicate hash
                        if current_file_size > 0:
                            self.skipped_by_hash_size += current_file_size
                    
                    pbar.update(1)
                    continue
                
                # Track size and count by type (standard categorization, even if moved by rule, original type is informative)
                # Size for actual copy will be handled by self.total_size if copy_file is successful
                if self.is_picture_file(file_path):
                    self.picture_count += 1 # Counts files intended for picture category
                    if not custom_move_folder: self.picture_size += current_file_size
                elif self.is_video_file(file_path):
                    self.video_count += 1
                    if not custom_move_folder: self.video_size += current_file_size
                else:
                    self.other_count += 1
                    if not custom_move_folder: self.other_size += current_file_size
                    
                if self.copy_file(file_path, target_path):
                    self.copied_files += 1
                    self.total_size += current_file_size # This is total size of *copied* files
                    self.already_copied_files.add(str(file_path))
                    if custom_move_folder: # If it was a rule-based move that was successful
                        self.moved_by_pattern_count += 1
                        self.moved_by_pattern_size += current_file_size
                    
                    # Check limit after successful copy
                    if self.limit and self.copied_files >= self.limit:
                        logger.info(f"Reached copy limit of {self.limit} files.")
                        pbar.update(1) # Ensure progress bar reflects this last file
                        limit_reached_flag = True
                        break # Exit the loop over files_to_process
                else:
                    self.skipped_files += 1 # Failed copy
                    self.skipped_size += current_file_size
                pbar.update(1)
                # Save log less frequently to reduce I/O
                if self.copied_files % 10 == 0:
                    self._save_log()
            # This inner loop is finished, pbar context will also exit.
            # No need to check limit_reached_flag to break this specific loop as it's already done.
        
        # If the limit was reached in the tqdm loop, we might want to stop processing further batches 
        # from the file_hashes.values() loop if that's how files_to_process was populated
        # However, files_to_process is now populated fully first, then iterated.
        # So, the break inside the tqdm loop is sufficient to stop processing more files than the limit.

        # Final log save
        self._save_log()
        
        logger.info(f"\nOrganization complete:")
        logger.info(f"Total source files: {self.total_source_files}")
        logger.info(f"Total source size: {self._format_size(self.total_source_size)}")
        logger.info(f"Files in cache: {total_files_seen}")
        logger.info(f"Files not in cache: {self.files_not_in_cache} files, {self._format_size(self.size_not_in_cache)}")
        logger.info(f"Files processed: {self.total_files}")
        logger.info(f"Files copied: {self.copied_files}")
        logger.info(f"Files skipped: {self.skipped_files}")
        logger.info(f"Files skipped by hash (duplicates): {self.skipped_by_hash} files, {self._format_size(self.skipped_by_hash_size)}")
        logger.info(f"Files skipped (no hash): {self.skipped_no_hash} files, {self._format_size(self.skipped_no_hash_size)}")
        logger.info(f"Files skipped (already copied): {self.skipped_already_copied} files, {self._format_size(self.skipped_already_copied_size)}")
        logger.info(f"Files skipped (not exists): {self.skipped_not_exists} files, {self._format_size(self.skipped_not_exists_size)}")
        logger.info(f"Files skipped by custom pattern: {self.skipped_by_pattern_count} files, {self._format_size(self.skipped_by_pattern_size)}")
        logger.info(f"\nTotal size to copy: {self._format_size(self.total_size)}")
        logger.info(f"  Pictures (standard location): {self.picture_count - self.moved_by_pattern_count if hasattr(self, 'moved_by_pattern_count') else self.picture_count} files, {self._format_size(self.picture_size)}") # Adjust count if moved
        logger.info(f"  Videos (standard location): {self.video_count} files, {self._format_size(self.video_size)}") # Similar adjustment might be needed
        logger.info(f"  Other (standard location): {self.other_count} files, {self._format_size(self.other_size)}") # Similar adjustment might be needed
        logger.info(f"Files moved by custom pattern: {self.moved_by_pattern_count} files, {self._format_size(self.moved_by_pattern_size)}")
        logger.info(f"Copies log saved to: {self.log_file}")

    def _format_size(self, size_bytes: int) -> str:
        """Convert bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"

    def _save_log(self):
        """Save the current state of copies to the log file."""
        try:
            with open(self.log_file, 'w') as f:
                json.dump({
                    'start_time': self.start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'total_files_processed': getattr(self, 'total_files', 0),
                    'files_copied': getattr(self, 'copied_files', 0),
                    'files_skipped': getattr(self, 'skipped_files', 0),
                    'total_size_copied': getattr(self, 'total_size', 0),
                    'copies': self.copies_log,
                    'dry_run': self.dry_run,
                    'source_dir': str(self.source_dir),
                    'target_dir': str(self.target_dir)
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving log file: {e}")

    @staticmethod
    def revert_copies(log_file: str, dry_run: bool = True):
        """Revert copies by removing the copied files."""
        logger.info(f"Reverting copies from log file: {log_file}")
        if dry_run:
            logger.info("Running in dry-run mode - no files will be deleted")
        
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            copies = log_data['copies']
            total_copies = len(copies)
            reverted = 0
            
            for copy in copies:
                target = Path(copy['target'])
                
                if not dry_run:
                    if target.exists():
                        # Delete the copied file
                        target.unlink()
                        reverted += 1
                        
                        if reverted % 100 == 0:
                            logger.info(f"Reverted {reverted} files...")
                    else:
                        logger.warning(f"Target file not found: {target}")
                else:
                    reverted += 1
                    if reverted % 100 == 0:
                        logger.info(f"Would revert {reverted} files...")
            
            logger.info(f"\nRevert complete:")
            logger.info(f"Total copies in log: {total_copies}")
            logger.info(f"Files reverted: {reverted}")
            
        except Exception as e:
            logger.error(f"Error reverting copies: {e}")

def main():
    parser = argparse.ArgumentParser(description='Organize backup files into media and document categories')
    parser.add_argument('source_dir', help='Source directory containing the files to organize')
    parser.add_argument('target_dir', help='Target directory for organized files')
    parser.add_argument('--dry-run', action='store_true', default=False,
                      help='Run in dry-run mode (default: False)')
    parser.add_argument('--cache', help='Path to the analyzer cache file')
    parser.add_argument('--limit', type=int, help='Limit the number of files to process')
    parser.add_argument('--revert', help='Revert copies from a log file')
    
    args = parser.parse_args()
    
    if args.revert:
        BackupOrganizer.revert_copies(args.revert, dry_run=args.dry_run)
    else:
        organizer = BackupOrganizer(args.source_dir, args.target_dir, dry_run=args.dry_run, 
                                  cache_file=args.cache, limit=args.limit)
        organizer.organize_files()

if __name__ == "__main__":
    main() 
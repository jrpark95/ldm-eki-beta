#!/usr/bin/env python3
"""
LDM-EKI Cleanup Script

Safely removes temporary data and shared memory files:
- ./logs/ directory contents
- ./output/ directory contents
- /dev/shm/ldm_eki_* shared memory files

Usage:
    python3 util/cleanup.py [options]

Options:
    --dry-run    Show what would be deleted without actually deleting
    --no-confirm Skip confirmation prompt
    --logs-only  Only clean logs directory
    --output-only Only clean output directory
    --shm-only   Only clean shared memory
"""

import os
import sys
import glob
import shutil
import argparse
from pathlib import Path

# ANSI color codes
class Color:
    RESET = '\033[0m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    BOLD = '\033[1m'

# Change to project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

# Known shared memory file patterns (from code analysis)
SHM_PATTERNS = [
    "/dev/shm/ldm_eki_*",
    "/dev/shm/ldm_eki_config",
    "/dev/shm/ldm_eki_data",
    "/dev/shm/ldm_eki_ensemble_config",
    "/dev/shm/ldm_eki_ensemble_data",
    "/dev/shm/ldm_eki_ensemble_obs_config",
    "/dev/shm/ldm_eki_ensemble_obs_data",
    "/dev/shm/ldm_eki_full_config",
]

def get_directory_size(path):
    """Calculate total size of a directory in bytes"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += get_directory_size(entry.path)
    except (PermissionError, FileNotFoundError):
        pass
    return total

def format_size(bytes_size):
    """Format bytes to human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def scan_shared_memory():
    """Scan for all LDM-EKI shared memory files"""
    shm_files = set()

    # Use glob patterns to find files
    for pattern in SHM_PATTERNS:
        matches = glob.glob(pattern)
        shm_files.update(matches)

    # Also scan /dev/shm directly for any ldm_eki files
    try:
        if os.path.exists('/dev/shm'):
            for entry in os.scandir('/dev/shm'):
                if entry.name.startswith('ldm_eki'):
                    shm_files.add(entry.path)
    except PermissionError:
        print("Warning: No permission to scan /dev/shm")

    return sorted(shm_files)

def count_files_recursive(path):
    """Count files recursively in a directory"""
    count = 0
    try:
        for root, dirs, files in os.walk(path):
            count += len(files)
    except (PermissionError, FileNotFoundError):
        pass
    return count

def clean_directory(path, dry_run=False):
    """Clean all contents of a directory while preserving the directory itself"""
    if not os.path.exists(path):
        print(f"  Directory does not exist: {path}")
        return 0, 0

    file_count = count_files_recursive(path)
    dir_size = get_directory_size(path)

    if file_count == 0:
        print(f"  ✓ Directory is already empty: {path}")
        return 0, 0

    if dry_run:
        print(f"  [DRY RUN] Would delete {file_count} files ({format_size(dir_size)}) from {path}")
        return file_count, dir_size

    # Delete contents
    removed_files = 0
    removed_size = 0

    try:
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    size = os.path.getsize(item_path)
                    os.unlink(item_path)
                    removed_files += 1
                    removed_size += size
                elif os.path.isdir(item_path):
                    size = get_directory_size(item_path)
                    shutil.rmtree(item_path)
                    removed_files += count_files_recursive(item_path)
                    removed_size += size
            except Exception as e:
                print(f"  Warning: Failed to delete {item_path}: {e}")

        print(f"  {Color.GREEN}✓{Color.RESET} Cleaned {path}: {Color.BOLD}{removed_files}{Color.RESET} files ({format_size(removed_size)})")
    except Exception as e:
        print(f"  Error cleaning {path}: {e}")
        return 0, 0

    return removed_files, removed_size

def clean_shared_memory(dry_run=False):
    """Clean shared memory files"""
    shm_files = scan_shared_memory()

    if not shm_files:
        print(f"  {Color.GREEN}✓{Color.RESET} No shared memory files found")
        return 0, 0

    removed_count = 0
    removed_size = 0

    for shm_file in shm_files:
        try:
            size = os.path.getsize(shm_file)

            if dry_run:
                print(f"  [DRY RUN] Would delete: {shm_file} ({format_size(size)})")
            else:
                os.unlink(shm_file)
                print(f"  {Color.GREEN}✓{Color.RESET} Deleted: {shm_file} ({format_size(size)})")
                removed_count += 1
                removed_size += size
        except FileNotFoundError:
            # File already deleted
            pass
        except PermissionError:
            print(f"  Warning: Permission denied: {shm_file}")
        except Exception as e:
            print(f"  Warning: Failed to delete {shm_file}: {e}")

    if not dry_run and removed_count > 0:
        print(f"  {Color.GREEN}✓{Color.RESET} Cleaned {Color.BOLD}{removed_count}{Color.RESET} shared memory files ({format_size(removed_size)})")

    return removed_count, removed_size

def main():
    parser = argparse.ArgumentParser(
        description='Clean LDM-EKI temporary data and shared memory files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be deleted without actually deleting')
    parser.add_argument('--no-confirm', action='store_true',
                        help='Skip confirmation prompt')
    parser.add_argument('--logs-only', action='store_true',
                        help='Only clean logs directory')
    parser.add_argument('--output-only', action='store_true',
                        help='Only clean output directory')
    parser.add_argument('--shm-only', action='store_true',
                        help='Only clean shared memory')

    args = parser.parse_args()

    # Determine what to clean
    clean_logs = args.logs_only or (not args.output_only and not args.shm_only)
    clean_output = args.output_only or (not args.logs_only and not args.shm_only)
    clean_shm = args.shm_only or (not args.logs_only and not args.output_only)

    print("=" * 70)
    print(f"{Color.CYAN}{Color.BOLD}LDM-EKI CLEANUP SCRIPT{Color.RESET}")
    print("=" * 70)
    print(f"Working directory: {Color.BOLD}{os.getcwd()}{Color.RESET}")

    if args.dry_run:
        print("\nDRY RUN MODE - No files will be deleted\n")

    # Show what will be cleaned
    print(f"\n{Color.BOLD}Items to be cleaned:{Color.RESET}")
    if clean_logs:
        print("  - ./logs/")
    if clean_output:
        print("  - ./output/")
    if clean_shm:
        print("  - /dev/shm/ldm_eki_* (shared memory)")

    # Scan and show preview
    print("\n" + "=" * 70)
    print(f"{Color.CYAN}SCANNING...{Color.RESET}")
    print("=" * 70)

    total_files = 0
    total_size = 0

    if clean_logs and os.path.exists('logs'):
        log_files = count_files_recursive('logs')
        log_size = get_directory_size('logs')
        print(f"logs/: {log_files} files, {format_size(log_size)}")
        total_files += log_files
        total_size += log_size

    if clean_output and os.path.exists('output'):
        output_files = count_files_recursive('output')
        output_size = get_directory_size('output')
        print(f"output/: {output_files} files, {format_size(output_size)}")
        total_files += output_files
        total_size += output_size

    if clean_shm:
        shm_files = scan_shared_memory()
        shm_size = sum(os.path.getsize(f) for f in shm_files if os.path.exists(f))
        print(f"shared memory: {len(shm_files)} files, {format_size(shm_size)}")
        total_files += len(shm_files)
        total_size += shm_size

    print(f"\n{Color.BOLD}Total: {total_files} files, {format_size(total_size)}{Color.RESET}")

    if total_files == 0:
        print(f"\n{Color.GREEN}✓{Color.RESET} Nothing to clean - all directories are already empty")
        return 0

    # Confirmation
    if not args.no_confirm and not args.dry_run:
        print("\n" + "=" * 70)
        response = input("Do you want to proceed? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("Aborted.")
            return 0

    # Perform cleanup
    print("\n" + "=" * 70)
    print(f"{Color.CYAN}CLEANING...{Color.RESET}")
    print("=" * 70)

    removed_files = 0
    removed_size = 0

    if clean_logs:
        print("\nCleaning logs/...")
        count, size = clean_directory('logs', args.dry_run)
        removed_files += count
        removed_size += size

    if clean_output:
        print("\nCleaning output/...")
        count, size = clean_directory('output', args.dry_run)
        removed_files += count
        removed_size += size

    if clean_shm:
        print("\nCleaning shared memory...")
        count, size = clean_shared_memory(args.dry_run)
        removed_files += count
        removed_size += size

    # Summary
    print("\n" + "=" * 70)
    print(f"{Color.CYAN}SUMMARY{Color.RESET}")
    print("=" * 70)

    if args.dry_run:
        print(f"Would delete: {removed_files} files ({format_size(removed_size)})")
        print("\nRun without --dry-run to actually delete files")
    else:
        print(f"Deleted: {removed_files} files ({format_size(removed_size)})")
        print(f"{Color.GREEN}✓{Color.RESET} Cleanup complete")

    print("=" * 70)
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

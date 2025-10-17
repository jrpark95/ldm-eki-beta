#!/usr/bin/env python3
"""
Memory Doctor - IPC Communication Debugging Utility

This module provides comprehensive logging and verification of all data transfers
between Python and C++ via POSIX shared memory. Used for debugging IPC issues
and verifying data integrity during EKI optimization.

Main Features:
    - Logs all data sent from Python to LDM
    - Logs all data received by Python from LDM
    - Computes checksums for data integrity verification
    - Tracks iteration numbers for proper data flow analysis
    - Automatic log directory cleanup on startup

Usage:
    Enable via MEMORY_DOCTOR_MODE=On in input/eki_settings.txt
    All logs are written to logs/memory_doctor/
    Format: iter{000}_py_{sent|recv}_{data_type}.txt

Author:
    Siho Jang, 2025

Examples:
    >>> from memory_doctor import memory_doctor
    >>> memory_doctor.set_enabled(True)
    >>> memory_doctor.log_sent_data("ensemble_states", states, iteration=1)
    >>> memory_doctor.log_received_data("observations", obs, iteration=1)
"""

import os
import sys
import glob
import numpy as np
from datetime import datetime
import struct
import hashlib

class MemoryDoctor:
    """
    IPC communication logging and verification utility.

    This class provides methods to log and verify all data transfers between
    Python and C++ during EKI optimization. Useful for debugging data corruption,
    dimension mismatches, and communication protocol issues.

    Attributes
    ----------
    enabled : bool
        Whether Memory Doctor logging is active
    log_dir : str
        Directory path for log files (default: "../../logs/memory_doctor/")

    Notes
    -----
    All log files follow the naming convention:
        iter{iteration:03d}_py_{sent|recv}_{data_type}.txt

    Each log file contains:
        - Data dimensions and statistics
        - MD5 checksum for integrity verification
        - Full data dump for manual inspection
        - Zero/NaN/Inf count analysis
    """

    def __init__(self):
        """Initialize Memory Doctor with logging disabled."""
        self.enabled = False
        self.log_dir = "../../logs/memory_doctor/"

    def set_enabled(self, enable):
        """
        Enable or disable Memory Doctor mode.

        Parameters
        ----------
        enable : bool
            True to enable logging, False to disable

        Notes
        -----
        When enabling, automatically:
            1. Creates log directory if it doesn't exist
            2. Cleans all previous log files
            3. Prints status message to console
        """
        self.enabled = enable
        if self.enabled:
            # Create log directory if it doesn't exist
            os.makedirs(self.log_dir, exist_ok=True)

            # Clean all existing log files
            self.clean_log_directory()

            print(f"[MEMORY_DOCTOR] ⚕️ Memory Doctor Mode ENABLED - All IPC data will be logged")
            print(f"[MEMORY_DOCTOR] 🧹 Cleaned previous log files from {self.log_dir}")

    def is_enabled(self):
        """
        Check if Memory Doctor logging is enabled.

        Returns
        -------
        bool
            True if logging is active, False otherwise
        """
        return self.enabled

    def clean_log_directory(self):
        """
        Remove all .txt log files from the log directory.

        Notes
        -----
        Called automatically when enabling Memory Doctor mode.
        Silently ignores errors during file deletion.
        """
        pattern = os.path.join(self.log_dir, "*.txt")
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
            except:
                pass  # Ignore errors during cleanup

    def calculate_checksum(self, data):
        """
        Calculate MD5 checksum for data integrity verification.

        Parameters
        ----------
        data : ndarray or array_like
            Data to checksum (numpy array or convertible to array)

        Returns
        -------
        str
            First 8 characters of MD5 hash in hexadecimal

        Notes
        -----
        Uses MD5 hash of the byte representation for fast checksumming.
        Truncated to 8 characters for compact logging.
        """
        if isinstance(data, np.ndarray):
            # Use hashlib for consistent checksum
            return hashlib.md5(data.tobytes()).hexdigest()[:8]
        else:
            # For lists or other iterables
            arr = np.array(data, dtype=np.float32)
            return hashlib.md5(arr.tobytes()).hexdigest()[:8]

    def log_received_data(self, data_type, data, iteration=0, extra_info=""):
        """
        Log data received by Python from C++ LDM.

        Parameters
        ----------
        data_type : str
            Type identifier for the data (e.g., "initial_observations", "ensemble_observations")
        data : ndarray or array_like
            Data array received from C++
        iteration : int, optional
            EKI iteration number (default: 0 for initial observations)
        extra_info : str, optional
            Additional information to include in log (default: "")

        Notes
        -----
        Log file is created as: iter{iteration:03d}_py_recv_{data_type}.txt

        Logged information includes:
            - Data dimensions and total element count
            - Min/max/mean statistics
            - MD5 checksum for verification
            - Zero/NaN/Inf/negative count
            - Complete data dump for inspection

        Examples
        --------
        >>> memory_doctor.log_received_data("initial_observations", obs, iteration=0)
        >>> memory_doctor.log_received_data("ensemble_observations", ens_obs, iteration=3)
        """
        if not self.enabled:
            return

        # Format: iter{000}_py_recv_{data_type}.txt
        filename = os.path.join(self.log_dir, f"iter{iteration:03d}_py_recv_{data_type}.txt")

        try:
            with open(filename, 'w') as f:
                f.write("=== MEMORY DOCTOR: PYTHON RECEIVED DATA ===\n")
                f.write(f"Iteration: {iteration}\n")
                f.write(f"Type: {data_type}\n")
                f.write(f"Direction: C++ → Python\n")

                # Handle different data types
                if isinstance(data, np.ndarray):
                    shape = data.shape
                    total = data.size
                    f.write(f"Dimensions: {' x '.join(map(str, shape))}\n")
                    f.write(f"Total Elements: {total}\n")
                    f.write(f"Dtype: {data.dtype}\n")

                    # Calculate statistics
                    checksum = self.calculate_checksum(data)
                    flat_data = data.flatten()

                    min_val = np.min(flat_data)
                    max_val = np.max(flat_data)
                    mean_val = np.mean(flat_data)
                    zero_count = np.sum(flat_data == 0)
                    nan_count = np.sum(np.isnan(flat_data))
                    inf_count = np.sum(np.isinf(flat_data))
                    neg_count = np.sum(flat_data < 0)

                else:
                    # Handle list or other iterable
                    flat_data = np.array(data).flatten()
                    shape = flat_data.shape
                    total = flat_data.size
                    f.write(f"Dimensions: {total}\n")
                    f.write(f"Total Elements: {total}\n")

                    checksum = self.calculate_checksum(flat_data)
                    min_val = np.min(flat_data)
                    max_val = np.max(flat_data)
                    mean_val = np.mean(flat_data)
                    zero_count = np.sum(flat_data == 0)
                    nan_count = np.sum(np.isnan(flat_data))
                    inf_count = np.sum(np.isinf(flat_data))
                    neg_count = np.sum(flat_data < 0)

                f.write(f"Checksum: {checksum}\n")
                f.write(f"Min: {min_val:.6e}\n")
                f.write(f"Max: {max_val:.6e}\n")
                f.write(f"Mean: {mean_val:.6e}\n")
                f.write(f"Zero Count: {zero_count} ({100.0 * zero_count / total:.2f}%)\n")
                f.write(f"Negative Count: {neg_count}\n")
                f.write(f"NaN Count: {nan_count}\n")
                f.write(f"Inf Count: {inf_count}\n")

                if extra_info:
                    f.write(f"Extra Info: {extra_info}\n")

                f.write("\n=== DATA (ALL ELEMENTS) ===\n")

                # Write all elements
                for i in range(len(flat_data)):
                    f.write(f"{flat_data[i]:12.6e} ")
                    if (i + 1) % 10 == 0:
                        f.write("\n")

                f.write("\n\n=== END OF DATA ===\n")

            print(f"[MEMORY_DOCTOR] 📥 Iteration {iteration}: Python received {data_type} ← C++")

        except Exception as e:
            print(f"[MEMORY_DOCTOR] Error logging received data: {e}")

    def log_sent_data(self, data_type, data, iteration=0, extra_info=""):
        """
        Log data sent from Python to C++ LDM.

        Parameters
        ----------
        data_type : str
            Type identifier for the data (e.g., "ensemble_states", "config")
        data : ndarray or array_like
            Data array being sent to C++
        iteration : int, optional
            EKI iteration number (default: 0 for initial data)
        extra_info : str, optional
            Additional information to include in log (default: "")

        Notes
        -----
        Log file is created as: iter{iteration:03d}_py_sent_{data_type}.txt

        Logged information includes:
            - Data dimensions and total element count
            - Min/max/mean statistics
            - MD5 checksum for verification
            - Zero/NaN/Inf/negative count
            - Complete data dump for inspection

        Examples
        --------
        >>> memory_doctor.log_sent_data("ensemble_states", states, iteration=1)
        >>> memory_doctor.log_sent_data("prior_ensemble", prior, iteration=0)
        """
        if not self.enabled:
            return

        # Format: iter{000}_py_sent_{data_type}.txt
        filename = os.path.join(self.log_dir, f"iter{iteration:03d}_py_sent_{data_type}.txt")

        try:
            with open(filename, 'w') as f:
                f.write("=== MEMORY DOCTOR: PYTHON SENT DATA ===\n")
                f.write(f"Iteration: {iteration}\n")
                f.write(f"Type: {data_type}\n")
                f.write(f"Direction: Python → C++\n")

                # Handle different data types
                if isinstance(data, np.ndarray):
                    shape = data.shape
                    total = data.size
                    f.write(f"Dimensions: {' x '.join(map(str, shape))}\n")
                    f.write(f"Total Elements: {total}\n")
                    f.write(f"Dtype: {data.dtype}\n")

                    # Calculate statistics
                    checksum = self.calculate_checksum(data)
                    flat_data = data.flatten()

                    min_val = np.min(flat_data)
                    max_val = np.max(flat_data)
                    mean_val = np.mean(flat_data)
                    zero_count = np.sum(flat_data == 0)
                    nan_count = np.sum(np.isnan(flat_data))
                    inf_count = np.sum(np.isinf(flat_data))
                    neg_count = np.sum(flat_data < 0)

                else:
                    # Handle list or other iterable
                    flat_data = np.array(data).flatten()
                    shape = flat_data.shape
                    total = flat_data.size
                    f.write(f"Dimensions: {total}\n")
                    f.write(f"Total Elements: {total}\n")

                    checksum = self.calculate_checksum(flat_data)
                    min_val = np.min(flat_data)
                    max_val = np.max(flat_data)
                    mean_val = np.mean(flat_data)
                    zero_count = np.sum(flat_data == 0)
                    nan_count = np.sum(np.isnan(flat_data))
                    inf_count = np.sum(np.isinf(flat_data))
                    neg_count = np.sum(flat_data < 0)

                f.write(f"Checksum: {checksum}\n")
                f.write(f"Min: {min_val:.6e}\n")
                f.write(f"Max: {max_val:.6e}\n")
                f.write(f"Mean: {mean_val:.6e}\n")
                f.write(f"Zero Count: {zero_count} ({100.0 * zero_count / total:.2f}%)\n")
                f.write(f"Negative Count: {neg_count}\n")
                f.write(f"NaN Count: {nan_count}\n")
                f.write(f"Inf Count: {inf_count}\n")

                if extra_info:
                    f.write(f"Extra Info: {extra_info}\n")

                f.write("\n=== DATA (ALL ELEMENTS) ===\n")

                # Write all elements
                for i in range(len(flat_data)):
                    f.write(f"{flat_data[i]:12.6e} ")
                    if (i + 1) % 10 == 0:
                        f.write("\n")

                f.write("\n\n=== END OF DATA ===\n")

            print(f"[MEMORY_DOCTOR] 📤 Iteration {iteration}: Python sent {data_type} → C++")

        except Exception as e:
            print(f"[MEMORY_DOCTOR] Error logging sent data: {e}")

# Global instance
memory_doctor = MemoryDoctor()
#!/usr/bin/env python3
"""
Memory Doctor for Python - IPC debugging utility
Logs all sent and received data for verification with proper EKI iteration tracking
"""

import os
import sys
import glob
import numpy as np
from datetime import datetime
import struct
import hashlib

class MemoryDoctor:
    def __init__(self):
        self.enabled = False
        self.log_dir = "../../logs/memory_doctor/"

    def set_enabled(self, enable):
        """Enable or disable Memory Doctor mode"""
        self.enabled = enable
        if self.enabled:
            # Create log directory if it doesn't exist
            os.makedirs(self.log_dir, exist_ok=True)

            # Clean all existing log files
            self.clean_log_directory()

            print(f"[MEMORY_DOCTOR] ⚕️ Memory Doctor Mode ENABLED - All IPC data will be logged")
            print(f"[MEMORY_DOCTOR] 🧹 Cleaned previous log files from {self.log_dir}")

    def is_enabled(self):
        return self.enabled

    def clean_log_directory(self):
        """Remove all .txt files in the log directory"""
        pattern = os.path.join(self.log_dir, "*.txt")
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
            except:
                pass  # Ignore errors during cleanup

    def calculate_checksum(self, data):
        """Calculate checksum for numpy array"""
        if isinstance(data, np.ndarray):
            # Use hashlib for consistent checksum
            return hashlib.md5(data.tobytes()).hexdigest()[:8]
        else:
            # For lists or other iterables
            arr = np.array(data, dtype=np.float32)
            return hashlib.md5(arr.tobytes()).hexdigest()[:8]

    def log_received_data(self, data_type, data, iteration=0, extra_info=""):
        """Log data received by Python from LDM with explicit iteration"""
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
        """Log data sent from Python to LDM with explicit iteration"""
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
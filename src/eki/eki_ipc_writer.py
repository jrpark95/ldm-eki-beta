"""
EKI IPC Writer - POSIX Shared Memory Writer for Python → LDM Communication

This module provides functionality to write ensemble state data from Python
to POSIX shared memory segments that will be read by the LDM-EKI C++ simulation.
"""

import os
import mmap
import struct
import numpy as np
from typing import Tuple
from memory_doctor import memory_doctor

# Color output support
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    # Fallback if colorama not available
    class DummyColor:
        def __getattr__(self, name):
            return ''
    Fore = Style = DummyColor()
    HAS_COLOR = False


class EKIIPCWriter:
    """
    POSIX Shared Memory Writer for ensemble state data communication.

    Writes ensemble state matrices from Python to shared memory segments
    that will be read by LDM-EKI C++ simulation.
    """

    # Shared memory names (must match C++ implementation)
    SHM_ENSEMBLE_CONFIG_NAME = "/ldm_eki_ensemble_config"
    SHM_ENSEMBLE_DATA_NAME = "/ldm_eki_ensemble_data"

    def __init__(self):
        self.config_written = False
        self.current_iteration = 0

    def write_ensemble_config(self, num_states: int, num_ensemble: int, iteration: int = 0) -> bool:
        """
        Write ensemble configuration to shared memory.

        Args:
            num_states: Number of state variables (e.g., 24 timesteps)
            num_ensemble: Number of ensemble members (e.g., 100)
            iteration: Current EKI iteration number (default 0)

        Returns:
            True if successful, False otherwise
        """
        config_path = f"/dev/shm{self.SHM_ENSEMBLE_CONFIG_NAME}"

        try:
            # Create or truncate config file
            with open(config_path, 'wb') as f:
                # Write 3 int32 values: num_states, num_ensemble, iteration
                config_data = struct.pack('<3i', num_states, num_ensemble, iteration)
                f.write(config_data)
                f.flush()
                os.fsync(f.fileno())

            # Set permissions
            os.chmod(config_path, 0o660)

            self.config_written = True
            self.current_iteration = iteration  # Store for later use
            return True

        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}[ERROR]{Style.RESET_ALL} Failed to write ensemble config: {e}")
            return False

    def write_ensemble_states(self, states: np.ndarray, num_states: int, num_ensemble: int) -> bool:
        """
        Write ensemble state matrix to shared memory.

        Args:
            states: 2D numpy array of shape (num_states, num_ensemble)
            num_states: Number of state variables (rows)
            num_ensemble: Number of ensemble members (columns)

        Returns:
            True if successful, False otherwise
        """
        if not self.config_written:
            print("[EKI_IPC_WRITER] Warning: Config not written, writing it now...")
            if not self.write_ensemble_config(num_states, num_ensemble):
                return False

        data_path = f"/dev/shm{self.SHM_ENSEMBLE_DATA_NAME}"

        try:
            # Validate dimensions
            if states.shape != (num_states, num_ensemble):
                raise ValueError(f"States shape {states.shape} doesn't match expected ({num_states}, {num_ensemble})")

            # Convert to float32 if necessary
            states_f32 = states.astype(np.float32, copy=False)

            # Calculate total size: header (12 bytes) + data
            header_size = 12  # 3 int32 values
            data_size = num_states * num_ensemble * 4  # float32
            total_size = header_size + data_size

            # Create file with proper size
            with open(data_path, 'wb') as f:
                # Write header: status=0 (writing), rows, cols
                header = struct.pack('<3i', 0, num_states, num_ensemble)
                f.write(header)

                # Write data in C order (row-major)
                # Python sends (num_states, num_ensemble) → C++ receives as rows x cols
                data_bytes = states_f32.tobytes(order='C')
                f.write(data_bytes)

                # Update status to 1 (ready)
                f.seek(0)
                header_ready = struct.pack('<3i', 1, num_states, num_ensemble)
                f.write(header_ready)

                f.flush()
                os.fsync(f.fileno())

            # Set permissions
            os.chmod(data_path, 0o660)

            # Memory Doctor: Log sent ensemble states with iteration
            if memory_doctor.is_enabled():
                iteration = getattr(self, 'current_iteration', 0)
                memory_doctor.log_sent_data("ensemble_states", states, iteration,
                                          f"Python->LDM ensemble states {num_states}x{num_ensemble} (iteration {iteration})")

            return True

        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}[ERROR]{Style.RESET_ALL} Failed to write ensemble states: {e}")
            print(f"  → Check /dev/shm permissions and available space")
            import traceback
            traceback.print_exc()
            return False

    @staticmethod
    def cleanup():
        """Remove shared memory files."""
        try:
            config_path = f"/dev/shm{EKIIPCWriter.SHM_ENSEMBLE_CONFIG_NAME}"
            data_path = f"/dev/shm{EKIIPCWriter.SHM_ENSEMBLE_DATA_NAME}"

            if os.path.exists(config_path):
                os.unlink(config_path)
            if os.path.exists(data_path):
                os.unlink(data_path)

            print("[EKI_IPC_WRITER] Shared memory cleaned up")
        except Exception as e:
            print(f"[EKI_IPC_WRITER] Cleanup failed: {e}")


def write_ensemble_to_shm(states: np.ndarray, num_states: int, num_ensemble: int) -> bool:
    """
    Convenience function to write ensemble states to shared memory.

    This function provides a simple interface to replace send_tmp_states().

    Args:
        states: 2D numpy array of shape (num_states, num_ensemble)
        num_states: Number of state variables (rows)
        num_ensemble: Number of ensemble members (columns)

    Returns:
        True if successful, False otherwise

    Example:
        >>> states = np.random.randn(24, 100)  # 24 timesteps, 100 ensemble members
        >>> success = write_ensemble_to_shm(states, 24, 100)
    """
    writer = EKIIPCWriter()

    # Write config first
    if not writer.write_ensemble_config(num_states, num_ensemble):
        return False

    # Write data
    if not writer.write_ensemble_states(states, num_states, num_ensemble):
        return False

    return True


if __name__ == "__main__":
    # Test the shared memory writer
    print("Testing EKI IPC Writer...")

    # Create test ensemble data: 24 states x 100 ensemble members
    num_states = 24
    num_ensemble = 100
    test_states = np.random.randn(num_states, num_ensemble).astype(np.float32) * 1e13

    print(f"Test data shape: {test_states.shape}")
    print(f"Test data range: [{test_states.min():.3e}, {test_states.max():.3e}]")

    # Write to shared memory
    success = write_ensemble_to_shm(test_states, num_states, num_ensemble)

    if success:
        print("\n✓ Test successful!")
        print("Shared memory files created:")
        print(f"  - /dev/shm/ldm_eki_ensemble_config")
        print(f"  - /dev/shm/ldm_eki_ensemble_data")
        print("\nYou can verify with: ls -lh /dev/shm/ldm_eki_ensemble_*")
    else:
        print("\n✗ Test failed!")

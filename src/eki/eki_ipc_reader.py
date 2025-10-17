"""
EKI IPC Reader - POSIX Shared Memory Reader for LDM-EKI Communication

This module provides functionality to read configuration and observation data
from POSIX shared memory segments created by the LDM-EKI C++ simulation.

The reader implements low-level binary data parsing for IPC communication
between the C++/CUDA forward model and the Python EKI inverse solver.

Shared Memory Segments
-----------------------
1. **Initial Configuration** (/dev/shm/ldm_eki_config)
   - Size: 12 bytes (basic) or 128 bytes (full config)
   - Format: Little-endian binary
   - Contents: Ensemble size, receptors, timesteps, algorithm parameters

2. **Initial Observations** (/dev/shm/ldm_eki_data)
   - Size: 12 bytes (header) + num_receptors × num_timesteps × 4 bytes (data)
   - Format: status (int32), rows (int32), cols (int32), data (float32[])
   - Contents: Gamma dose observations from reference simulation

3. **Ensemble Configuration** (/dev/shm/ldm_eki_ensemble_config)
   - Size: 12 bytes
   - Format: num_states (int32), num_ensemble (int32), iteration_id (int32)

4. **Ensemble Observations** (/dev/shm/ldm_eki_ensemble_obs_config, ..._data)
   - Config: 12 bytes (ensemble_size, num_receptors, num_timesteps)
   - Data: ensemble_size × num_receptors × num_timesteps × 4 bytes (float32)
   - Layout: [Ens0: T0_R0, T0_R1, ..., T1_R0, ...], [Ens1: ...]

Binary Data Formats
-------------------
All integers and floats are little-endian (<) format.

**Configuration Structure (128 bytes):**

    Offset | Size | Type     | Field
    -------|------|----------|------------------
    0      | 4    | int32    | ensemble_size
    4      | 4    | int32    | num_receptors
    8      | 4    | int32    | num_timesteps
    12     | 4    | int32    | iteration
    16     | 4    | float32  | renkf_lambda
    20     | 4    | float32  | noise_level
    24     | 4    | float32  | time_days
    28     | 4    | float32  | time_interval
    32     | 4    | float32  | inverse_time_interval
    36     | 4    | float32  | receptor_error
    40     | 4    | float32  | receptor_mda
    44     | 4    | float32  | prior_constant
    48     | 4    | int32    | num_source
    52     | 4    | int32    | num_gpu
    56     | 8    | char[8]  | perturb_option
    64     | 8    | char[8]  | adaptive_eki
    72     | 8    | char[8]  | localized_eki
    80     | 8    | char[8]  | regularization
    88     | 8    | char[8]  | gpu_forward
    96     | 8    | char[8]  | gpu_inverse
    104    | 8    | char[8]  | source_location
    112    | 8    | char[8]  | time_unit
    120    | 8    | char[8]  | memory_doctor

**Observation Data Structure:**

    Header (12 bytes):
        status (int32): 1 = ready, 0 = not ready
        rows (int32): Number of receptors
        cols (int32): Number of timesteps

    Data (rows × cols × 4 bytes):
        float32 array in row-major (C) order
        Layout: [R0_T0, R0_T1, ..., R0_T23, R1_T0, R1_T1, ...]

**Ensemble Observation Data:**

    Config (12 bytes):
        ensemble_size (int32)
        num_receptors (int32)
        num_timesteps (int32)

    Data (ensemble_size × num_timesteps × num_receptors × 4 bytes):
        float32 array in nested row-major order
        Layout: [Ens0: T0_R0, T0_R1, T0_R2, T1_R0, ...],
                [Ens1: T0_R0, T0_R1, T0_R2, T1_R0, ...], ...

Error Handling
--------------
- OSError: Raised when shared memory files cannot be accessed
- RuntimeError: Raised when data format is invalid or corrupted
- BufferError: Handled gracefully during memory map cleanup

Performance Notes
-----------------
- Memory mapping (mmap) used for efficient large data transfers
- Typical read times:
  - Initial observations (3 × 24): < 1 ms
  - Ensemble observations (100 × 24 × 3): < 10 ms
- Zero-copy operations where possible using memoryview

Usage Example
-------------
>>> from eki_ipc_reader import EKIIPCReader, receive_gamma_dose_matrix_shm
>>>
>>> # Read initial observations
>>> obs_3d = receive_gamma_dose_matrix_shm()  # shape (1, 3, 24)
>>>
>>> # Read ensemble observations (during EKI iteration)
>>> from eki_ipc_reader import receive_ensemble_observations_shm
>>> ens_obs = receive_ensemble_observations_shm()  # shape (100, 24, 3)

Author
------
Siho Jang, 2025

References
----------
.. [1] POSIX.1-2008, "shm_open - open a shared memory object"
.. [2] Linux Programmer's Manual, "mmap(2) - map files or devices into memory"
"""

import os
import mmap
import struct
import time
import numpy as np
from typing import Tuple, Optional
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


class EKIIPCReader:
    """
    POSIX Shared Memory Reader for EKI data communication.
    
    Reads configuration and observation data from shared memory segments
    created by LDM-EKI C++ simulation.
    """
    
    # Shared memory names (must match C++ implementation)
    SHM_CONFIG_NAME = "/ldm_eki_config"
    SHM_DATA_NAME = "/ldm_eki_data"
    
    def __init__(self):
        self.ensemble_size = None
        self.num_receptors = None
        self.num_timesteps = None
        self._config_loaded = False
    
    def read_eki_config(self) -> Tuple[int, int, int]:
        """
        Read basic EKI configuration from shared memory.

        Reads the first 12 bytes of /dev/shm/ldm_eki_config containing
        the three fundamental parameters needed to allocate observation arrays.

        Returns
        -------
        tuple of int
            (ensemble_size, num_receptors, num_timesteps)
            e.g., (100, 3, 24) for 100 ensemble members, 3 receptors, 24 timesteps

        Raises
        ------
        OSError
            If shared memory file cannot be opened or read
        RuntimeError
            If file size is incorrect (not 12 bytes minimum)

        Notes
        -----
        **Binary Format:**
        - Offset 0-3: ensemble_size (int32, little-endian)
        - Offset 4-7: num_receptors (int32, little-endian)
        - Offset 8-11: num_timesteps (int32, little-endian)

        Total size: 12 bytes

        **Struct Format String:**
        '<3i' = little-endian (<), 3 signed integers (3i)

        This function must be called before read_eki_observations() to validate
        array dimensions.

        Examples
        --------
        >>> reader = EKIIPCReader()
        >>> ens_size, n_rec, n_time = reader.read_eki_config()
        >>> print(f"Config: {ens_size} ensembles, {n_rec} receptors, {n_time} timesteps")
        Config: 100 ensembles, 3 receptors, 24 timesteps

        Author
        ------
        Siho Jang, 2025
        """
        config_path = f"/dev/shm{self.SHM_CONFIG_NAME}"
        
        try:
            # Open and read config
            with open(config_path, 'rb') as f:
                # Read 3 int32 values: ensemble_size, num_receptors, num_timesteps
                data = f.read(12)  # 3 * sizeof(int32)
                if len(data) != 12:
                    raise RuntimeError(f"Invalid config data size: {len(data)} bytes")
                
                # Unpack little-endian int32 values
                self.ensemble_size, self.num_receptors, self.num_timesteps = struct.unpack('<3i', data)
                self._config_loaded = True

                return self.ensemble_size, self.num_receptors, self.num_timesteps

        except OSError as e:
            raise OSError(f"{Fore.RED}{Style.BRIGHT}[ERROR]{Style.RESET_ALL} Failed to read config shared memory: {e}\n"
                        f"  → Check if LDM has created /dev/shm/ldm_eki_config")
    
    def read_eki_observations(self) -> np.ndarray:
        """
        Read EKI observation data from shared memory.
        
        Returns:
            2D numpy array of shape (num_receptors, num_timesteps) with observation data
            
        Raises:
            RuntimeError: If config not loaded
            OSError: If shared memory access fails
        """
        if not self._config_loaded:
            raise RuntimeError("Config must be loaded first using read_eki_config()")
        
        data_path = f"/dev/shm{self.SHM_DATA_NAME}"
        
        try:
            # Open data file
            fd = os.open(data_path, os.O_RDONLY)
            
            try:
                # Get file size
                st = os.fstat(fd)
                total_bytes = st.st_size
                
                # Map entire file
                mm = mmap.mmap(fd, length=total_bytes, access=mmap.ACCESS_READ)
                
                try:
                    # Read header: status, rows, cols (3 int32 values)
                    header_data = mm[:12]
                    status, rows, cols = struct.unpack('<3i', header_data)
                    
                    if status != 1:
                        raise RuntimeError(f"Data not ready (status={status})")
                    
                    # Verify dimensions
                    if rows != self.num_receptors or cols != self.num_timesteps:
                        raise RuntimeError(f"Dimension mismatch: expected {self.num_receptors}x{self.num_timesteps}, "
                                         f"got {rows}x{cols}")
                    
                    # Read observation data
                    data_offset = 12  # After header
                    data_count = rows * cols
                    
                    # Create numpy array from memory view
                    mv = memoryview(mm)[data_offset:data_offset + data_count * 4]
                    observations = np.frombuffer(mv, dtype=np.float32, count=data_count)
                    
                    # Reshape to (receptors, timesteps) - row-major format
                    observations_2d = observations.reshape((rows, cols), order='C')

                    result = observations_2d.copy()  # Copy to ensure data persists after mmap close
                    
                finally:
                    try:
                        mm.close()
                    except BufferError:
                        pass  # Ignore buffer error, data already copied
                
                return result
            finally:
                os.close(fd)
                
        except OSError as e:
            raise OSError(f"Failed to read observation shared memory: {e}")
    
    def get_config(self) -> Optional[Tuple[int, int, int]]:
        """
        Get currently loaded configuration without reading from shared memory.
        
        Returns:
            Tuple of (ensemble_size, num_receptors, num_timesteps) if config loaded, None otherwise
        """
        if self._config_loaded:
            return self.ensemble_size, self.num_receptors, self.num_timesteps
        return None
    
    def is_config_loaded(self) -> bool:
        """Check if configuration has been loaded."""
        return self._config_loaded


def receive_gamma_dose_matrix_shm() -> np.ndarray:
    """
    Convenience function to replace receive_gamma_dose_matrix() with shared memory version.
    
    This function provides the same interface as the original TCP-based function
    but uses POSIX shared memory for much better performance.
    
    Returns:
        3D numpy array of shape (1, num_receptors, num_timesteps) to match original interface
        
    Raises:
        RuntimeError: If data cannot be read
        OSError: If shared memory access fails
    """
    reader = EKIIPCReader()
    
    # Read config first
    ensemble_size, num_receptors, num_timesteps = reader.read_eki_config()
    
    # Read observations
    observations_2d = reader.read_eki_observations()
    
    # Add batch dimension to match original interface: (1, receptors, timesteps)
    observations_3d = np.expand_dims(observations_2d, axis=0)

    # Memory Doctor: Log received initial observations (iteration 0)
    if memory_doctor.is_enabled():
        memory_doctor.log_received_data("initial_observations", observations_2d, 0,
                                       f"LDM->Python initial observations {num_receptors}x{num_timesteps}")

    return observations_3d


def read_eki_full_config_shm() -> dict:
    """
    Read full EKI configuration from shared memory (80 bytes).

    This function reads the extended configuration structure that includes
    all Python EKI parameters, not just the basic 3 integers.

    Structure (v1.0 Release - Simplified):
    - Bytes 0-11  : Basic dimensions (12 bytes)
    - Bytes 12-31 : Algorithm parameters (20 bytes)
    - Bytes 32-71 : Option strings (40 bytes = 5 strings × 8 bytes)
    - Bytes 72-79 : Memory Doctor mode (8 bytes)

    Returns:
        dict: Dictionary with all configuration parameters

    Raises:
        OSError: If shared memory access fails
        RuntimeError: If data size is incorrect
    """
    config_path = "/dev/shm/ldm_eki_config"

    try:
        with open(config_path, 'rb') as f:
            # Read full config structure (80 bytes)
            data = f.read(80)
            if len(data) < 80:  # Minimum 80 bytes for actual data
                raise RuntimeError(f"Invalid full config data size: {len(data)} bytes (expected 80)")

            # Unpack structure (little-endian):
            # Basic info (12 bytes): 3 int32
            # Algorithm params (20 bytes): 1 int32 + 4 float32
            # Strings (40 bytes): 5 char[8]
            # Memory Doctor (8 bytes): 1 char[8]
            # Total: 12 + 20 + 40 + 8 = 80 bytes

            # Basic info (12 bytes)
            ensemble_size, num_receptors, num_timesteps = struct.unpack_from('<3i', data, 0)

            # Algorithm parameters (20 bytes starting at offset 12)
            offset = 12
            iteration, = struct.unpack_from('<i', data, offset)
            offset += 4
            renkf_lambda, noise_level, time_interval, prior_constant = struct.unpack_from('<4f', data, offset)
            offset += 16  # Now at 32 bytes

            # Option strings (40 bytes = 5 × 8 bytes, starting at offset 32)
            perturb_option = data[32:40].decode('utf-8').rstrip('\x00')
            adaptive_eki = data[40:48].decode('utf-8').rstrip('\x00')
            localized_eki = data[48:56].decode('utf-8').rstrip('\x00')
            regularization = data[56:64].decode('utf-8').rstrip('\x00')
            time_unit = data[64:72].decode('utf-8').rstrip('\x00')

            # Memory Doctor Mode (8 bytes, starting at offset 72)
            memory_doctor = data[72:80].decode('utf-8').rstrip('\x00')

            config_dict = {
                # Basic
                'ensemble_size': ensemble_size,
                'num_receptors': num_receptors,
                'num_timesteps': num_timesteps,

                # Algorithm
                'iteration': iteration,
                'renkf_lambda': renkf_lambda,
                'noise_level': noise_level,
                'time_interval': time_interval,
                'prior_constant': prior_constant,  # Prior emission constant value

                # Options
                'perturb_option': perturb_option,
                'adaptive_eki': adaptive_eki,
                'localized_eki': localized_eki,
                'regularization': regularization,
                'time_unit': time_unit,
                'memory_doctor': memory_doctor,  # Memory Doctor mode setting

                # Hardcoded values (v1.0 release - not in IPC structure)
                'GPU_ForwardPhysicsModel': 'On',  # Always enabled (CUDA required)
                'GPU_InverseModel': 'On',  # Always enabled (CuPy required)
                'nGPU': 1,  # Single GPU only
                'Source_location': 'Fixed',  # Known position (estimate emissions only)
                'nsource': 1,  # Single source only
            }

            return config_dict

    except OSError as e:
        raise OSError(f"Failed to read full config shared memory: {e}")


if __name__ == "__main__":
    # Test the shared memory reader
    print("Testing EKI IPC Reader...")

    try:
        # Test full config reading
        print("\n=== Testing Full Config Reading ===")
        full_config = read_eki_full_config_shm()
        print(f"Full configuration loaded: {len(full_config)} parameters")
        for key, value in full_config.items():
            print(f"  {key}: {value}")

        print("\n=== Testing Basic Config Reading ===")
        reader = EKIIPCReader()
        config = reader.read_eki_config()
        print(f"Basic configuration: {config}")

        # Test observation reading
        print("\n=== Testing Observation Reading ===")
        observations = reader.read_eki_observations()
        print(f"Observations shape: {observations.shape}")
        print(f"Sample data: {observations[0, :5]}")  # First receptor, first 5 timesteps

        # Test convenience function
        print("\n=== Testing Convenience Function ===")
        gamma_data = receive_gamma_dose_matrix_shm()
        print(f"Gamma dose data shape: {gamma_data.shape}")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# Functions for receiving ensemble observations from LDM
# =============================================================================

def receive_ensemble_observations_shm(current_iteration=None):
    """
    Receive ensemble observation data from LDM via shared memory.

    Reads predicted observations for all ensemble members after LDM completes
    ensemble simulations. This is the core data transfer in each EKI iteration.

    Parameters
    ----------
    current_iteration : int, optional
        Current EKI iteration number (used for Memory Doctor logging)

    Returns
    -------
    observations : ndarray, shape (num_ensemble, num_timesteps, num_receptors)
        3D array of ensemble observations
        - observations[i, t, r] = dose at receptor r, time t, ensemble i
        - Typical shape: (100, 24, 3) for 100 members, 24 timesteps, 3 receptors

    Raises
    ------
    FileNotFoundError
        If shared memory files /dev/shm/ldm_eki_ensemble_obs_* don't exist
    RuntimeError
        If configuration dimensions don't match expected values
    OSError
        If memory mapping fails

    Notes
    -----
    **Binary Data Format:**

    Configuration file (/dev/shm/ldm_eki_ensemble_obs_config):
        - Offset 0-3: ensemble_size (int32)
        - Offset 4-7: num_receptors (int32)
        - Offset 8-11: num_timesteps (int32)
        - Total: 12 bytes

    Data file (/dev/shm/ldm_eki_ensemble_obs_data):
        - float32 array in row-major (C) order
        - Memory layout: [Ens0: T0_R0, T0_R1, T0_R2, T1_R0, ...],
                         [Ens1: T0_R0, T0_R1, T0_R2, T1_R0, ...], ...
        - Total size: ensemble_size × num_timesteps × num_receptors × 4 bytes

    **Memory Layout Example:**

    For 2 ensembles, 3 timesteps, 2 receptors:

        Raw bytes (little-endian float32):
        [Ens0_T0_R0, Ens0_T0_R1, Ens0_T1_R0, Ens0_T1_R1, Ens0_T2_R0, Ens0_T2_R1,
         Ens1_T0_R0, Ens1_T0_R1, Ens1_T1_R0, Ens1_T1_R1, Ens1_T2_R0, Ens1_T2_R1]

        Reshaped to (2, 3, 2):
        observations[0, :, :] = [[Ens0_T0_R0, Ens0_T0_R1],
                                 [Ens0_T1_R0, Ens0_T1_R1],
                                 [Ens0_T2_R0, Ens0_T2_R1]]

        observations[1, :, :] = [[Ens1_T0_R0, Ens1_T0_R1],
                                 [Ens1_T1_R0, Ens1_T1_R1],
                                 [Ens1_T2_R0, Ens1_T2_R1]]

    **Performance:**
    - Memory mapping for efficient zero-copy read
    - Typical read time: < 10 ms for 100 × 24 × 3 = 7200 float32 values
    - Data copied to ensure persistence after mmap close

    **Memory Doctor Mode:**
    If enabled, saves received data to /tmp/eki_debug/ for debugging.

    Examples
    --------
    >>> # Called by Model.state_to_ob() after LDM simulation completes
    >>> obs = receive_ensemble_observations_shm(iteration=5)
    >>> print(f"Received {obs.shape[0]} ensemble observations")
    Received 100 ensemble observations
    >>> print(f"Shape: {obs.shape} (ensemble, timestep, receptor)")
    Shape: (100, 24, 3) (ensemble, timestep, receptor)
    >>> # Check data validity
    >>> print(f"Min: {obs.min():.3e}, Max: {obs.max():.3e}")
    Min: 0.000e+00, Max: 1.234e-10

    See Also
    --------
    receive_gamma_dose_matrix_shm : Read initial observations (reference simulation)
    EKIIPCWriter.write_ensemble_states : Corresponding writer function

    Author
    ------
    Siho Jang, 2025
    """
    import mmap
    import struct
    import numpy as np

    # Shared memory file paths
    config_path = "/dev/shm/ldm_eki_ensemble_obs_config"
    data_path = "/dev/shm/ldm_eki_ensemble_obs_data"

    try:
        # Read configuration first
        with open(config_path, 'rb') as f:
            config_data = f.read(12)
            if len(config_data) < 12:
                raise RuntimeError(f"Invalid ensemble obs config size: {len(config_data)} bytes")

            # Unpack config: 3 int32 values (little-endian)
            ensemble_size, num_receptors, num_timesteps = struct.unpack('<3i', config_data)

        # Read observation data
        with open(data_path, 'rb') as f:
            # Memory map for efficient reading
            mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            # Calculate expected size
            expected_size = ensemble_size * num_receptors * num_timesteps * 4  # float32

            # Read all data
            data = mmapped_file.read(expected_size)

            # Convert to numpy array
            flat_array = np.frombuffer(data, dtype=np.float32)

            # Reshape to 3D array [ensemble, timestep, receptor]
            # C++ sends memory as: [Ens0: T0_R0, T0_R1, T0_R2, T1_R0, ...]
            # Binary reshape MUST match C++ memory order
            observations = flat_array.reshape(ensemble_size, num_timesteps, num_receptors)

            # Close memory map
            mmapped_file.close()

            # Memory Doctor: Log received ensemble observations with iteration
            if memory_doctor.is_enabled():
                iteration = current_iteration if current_iteration is not None else 0
                memory_doctor.log_received_data("ensemble_observations", observations, iteration,
                                               f"LDM->Python ensemble obs {ensemble_size}x{num_receptors}x{num_timesteps} (iteration {iteration})")

            return observations

    except FileNotFoundError as e:
        print(f"{Fore.RED}{Style.BRIGHT}[ERROR]{Style.RESET_ALL} Ensemble observation files not found")
        print(f"  → Check if LDM has written to /dev/shm/ldm_eki_ensemble_obs_*")
        raise e
    except Exception as e:
        print(f"{Fore.RED}{Style.BRIGHT}[ERROR]{Style.RESET_ALL} Failed to read ensemble observations: {e}")
        raise e
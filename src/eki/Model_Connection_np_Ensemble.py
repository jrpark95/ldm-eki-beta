"""
Model Connection for Ensemble EKI (NumPy-based)

This module provides the interface between the Python EKI optimizer and the
LDM-EKI C++/CUDA forward model via POSIX shared memory IPC.

Main components:
    - Configuration loading from shared memory (replaces YAML files)
    - Initial observation reception from LDM
    - Ensemble state transmission to LDM
    - Ensemble observation reception from LDM
    - Forward model wrapper for state_to_ob() calls

Communication flow:
    1. LDM writes initial observations → Python reads via shared memory
    2. Python generates prior ensemble
    3. For each EKI iteration:
       a. Python writes ensemble states → LDM reads via shared memory
       b. LDM runs ensemble simulations
       c. LDM writes ensemble observations → Python reads via shared memory
       d. Python updates ensemble via Kalman gain

Array conventions:
    - Python: Column-major (num_states, num_ensemble)
    - C++: Row-major [ensemble][state]
    - Shared memory: Always row-major (flatten with order='C')

Author:
    Siho Jang, 2025

References:
    - POSIX shared memory: shm_open(3), mmap(2)
    - IPC data layout: See eki_ipc_reader.py and eki_ipc_writer.py
"""

import numpy as np
# import GaussianPuffClass_Rev_20240514 as Gpuff
# import GaussianPuffClass_Rev_20240514_GPU_Ensemble as Gpuff_Ensemble
# import pycuda.driver as cuda
# import pycuda.autoinit
import concurrent.futures
import multiprocessing
from multiprocessing import Process, Queue
from copy import deepcopy

import socket
import struct
from eki_ipc_reader import receive_gamma_dose_matrix_shm, EKIIPCReader, read_eki_full_config_shm
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

def load_config_from_shared_memory():
    """
    Load configuration from shared memory and convert to input_config/input_data dictionaries.

    This function replaces YAML file parsing by reading all parameters from shared memory
    (written by LDM C++) and constructing the same dictionary structures expected by the
    Python EKI code. It serves as the IPC bridge for configuration data.

    Returns
    -------
    tuple of dict
        (input_config, input_data) dictionaries matching YAML format

    Notes
    -----
    **Shared Memory Structure:**

    The configuration is stored in /dev/shm/ldm_eki_config (128 bytes):

    Byte Layout (little-endian):
    - Bytes 0-11 (12): Basic config (3 × int32)
        - ensemble_size, num_receptors, num_timesteps
    - Bytes 12-55 (44): Algorithm parameters
        - iteration (int32)
        - renkf_lambda, noise_level, time_days, time_interval,
          inverse_time_interval (5 × float32)
        - receptor_error, receptor_mda, prior_constant (3 × float32)
        - num_source, num_gpu (2 × int32)
    - Bytes 56-127 (72): Option strings (9 × char[8])
        - perturb_option, adaptive_eki, localized_eki, regularization,
          gpu_forward, gpu_inverse, source_location, time_unit, memory_doctor

    **Dictionary Construction:**

    input_config contains EKI algorithm parameters:
    - sample_ctrl: Ensemble size
    - iteration: Number of EKI iterations
    - Optimizer_order: List of algorithms to run ['EKI']
    - Adaptive_EKI, Localized_EKI, Regularization: On/Off flags
    - GPU_ForwardPhysicsModel, GPU_InverseModel: GPU usage flags
    - nGPU: Number of GPU devices

    input_data contains simulation parameters:
    - time: Simulation duration in days
    - nreceptor: Number of observation receptors
    - nsource: Number of emission sources
    - Source_1: True emission time series (for reference simulation)
    - Prior_Source_1: Prior emission guess (for inversion)
    - receptor_position: List of [lat, lon, alt] coordinates

    **Example:**
    >>> input_config, input_data = load_config_from_shared_memory()
    >>> print(f"Ensemble size: {input_config['sample_ctrl']}")
    Ensemble size: 100
    >>> print(f"Receptors: {input_data['nreceptor']}")
    Receptors: 3

    Raises
    ------
    OSError
        If shared memory file cannot be read

    Author
    ------
    Siho Jang, 2025
    """
    print(f"{Fore.MAGENTA}[ENSEMBLE]{Style.RESET_ALL} Loading configuration from shared memory...")

    # Read full configuration from shared memory
    shm_data = read_eki_full_config_shm()

    # Enable Memory Doctor mode if configured
    # FAIL-FAST VALIDATION: No .get() defaults - must be explicitly present in config
    if 'memory_doctor' not in shm_data:
        print(f"\033[91m\033[1m[INPUT ERROR]\033[0m Missing required parameter: memory_doctor")
        print()
        print(f"  \033[93mProblem:\033[0m")
        print(f"    The memory_doctor parameter is required in shared memory configuration.")
        print(f"    This controls debug logging of IPC data transfers between C++ and Python.")
        print()
        print(f"  \033[96mRequired value:\033[0m")
        print(f"    memory_doctor: 'On' or 'Off'")
        print()
        print(f"  \033[92mDebug features when 'On':\033[0m")
        print(f"    - Saves all ensemble states to /tmp/eki_debug/ensemble_states_*.dat")
        print(f"    - Saves all observations to /tmp/eki_debug/observations_*.dat")
        print(f"    - Enables detailed IPC transfer logging")
        print()
        print(f"  \033[92mRecommended:\033[0m")
        print(f"    - Development/debugging: 'On'")
        print(f"    - Production runs: 'Off' (saves disk I/O)")
        print()
        print(f"  \033[96mFix in:\033[0m input/eki.conf (must set MEMORY_DOCTOR_MODE)")
        import sys
        sys.exit(1)

    # Validate memory_doctor value
    memory_doctor_value = shm_data['memory_doctor'].strip()
    if memory_doctor_value.lower() not in ['on', 'off', '1', '0', 'true', 'false']:
        print(f"\033[91m\033[1m[INPUT ERROR]\033[0m Invalid memory_doctor value: '{memory_doctor_value}'")
        print()
        print(f"  \033[93mProblem:\033[0m")
        print(f"    memory_doctor must be 'On' or 'Off' (case-insensitive).")
        print()
        print(f"  \033[96mRequired value:\033[0m")
        print(f"    MEMORY_DOCTOR_MODE: <'On' | 'Off' | '1' | '0' | 'True' | 'False'>")
        print()
        print(f"  \033[92mExamples:\033[0m")
        print(f"    MEMORY_DOCTOR_MODE: Off  # Normal operation")
        print(f"    MEMORY_DOCTOR_MODE: On   # Debug mode")
        print()
        print(f"  \033[96mFix in:\033[0m input/eki.conf")
        import sys
        sys.exit(1)

    if memory_doctor_value.lower() in ['on', '1', 'true']:
        memory_doctor.set_enabled(True)
        print(f"{Fore.YELLOW}[DEBUG]{Style.RESET_ALL} ⚕️  Memory Doctor Mode enabled")
    else:
        memory_doctor.set_enabled(False)

    # Construct input_config dictionary (matches original YAML structure)
    input_config = {
        'sample_ctrl': shm_data['ensemble_size'],
        'nrepeat': 1,
        'time': 1,
        'iteration': shm_data['iteration'],
        'Optimizer_order': ['EKI'],

        # EKI options
        'perturb_option': shm_data['perturb_option'],
        'EnRML_step_length': 1.0,
        'EnKF_MDA_steps': 0.7,
        'REnKF_regularization': 'regularization.py',
        'Adaptive_EKI': shm_data['adaptive_eki'],
        'Localized_EKI': shm_data['localized_eki'],
        'Localization': 'centralized',
        'Localization_weighting_factor': 1.0,
        'Regularization': shm_data['regularization'],
        'REnKF_lambda': shm_data['renkf_lambda'],

        # Other options
        'Elimination': 'Off',
        'Elimination_condition': 1.0e+6,
        'Receptor_Increment': 'Off',

        # GPU configuration (v1.0: Hardcoded to always use GPU)
        'GPU_ForwardPhysicsModel': 'On',  # Always enabled (CUDA required)
        'GPU_InverseModel': 'On',          # Always use CuPy for inverse model
        'nGPU': 1,                         # Single GPU mode
    }

    # Construct input_data dictionary (matches original YAML structure)
    input_data = {
        # Time parameters
        'time': 0.25,  # Derived from simulation.conf time_end
        'time_interval': 15,  # Fixed: 15 minutes (from eki.conf)
        'inverse_time_interval': 0.25,  # Derived: time_interval converted to hours
        'ave_t': 3,  # Fixed: 15 min / 5 min = 3

        # Grid parameters (default values - not used in LDM forward model)
        'size_alt': 100,
        'size_lat': 100,
        'size_lon': 100,
        'alt_spacing': 500.0,
        'lat_spacing': 500.0,
        'lon_spacing': 500.0,

        # Wind parameters (not used in LDM forward model)
        'size_alt_wind': 100,
        'size_lat_wind': 100,
        'size_lon_wind': 100,
        'wind_init_mode': None,
        'wind_constant_value_x': 2.50,
        'wind_constant_value_y': 0.0,
        'wind_constant_value_z': 0.0,
        'wind_grid_interval': 1500,
        'grid_space_size_lat': 100,
        'grid_space_size_lon': 100,

        # Puff parameters (not used in LDM forward model)
        'puff_concentration_threshold': 0.0e-10,
        'R_max': 1.0E+200,

        # Receptor parameters
        'nreceptor': shm_data['num_receptors'],
        'receptor_position': [[1000.0, 1000.0, 1.0], [2000.0, 2000.0, 1.0], [3000.0, 3000.0, 1.0],
                              [4000.0, 4000.0, 1.0], [5000.0, 5000.0, 1.0], [6000.0, 6000.0, 1.0],
                              [7000.0, 7000.0, 1.0], [8000.0, 8000.0, 1.0], [9000.0, 9000.0, 1.0],
                              [10000.0, 10000.0, 1.0], [12500.0, 12500.0, 1.0], [15000.0, 15000.0, 1.0]],
        'nreceptor_err': 0.0,  # No additional measurement error (noise in observations)
        'nreceptor_MDA': 0.0,  # No MDA inflation

        # Source parameters (v1.0: Hardcoded to Fixed location)
        'Source_location': 'Fixed',  # Always use known source position
        'nsource': 1,                # Always single source

        # Calculate number of state timesteps dynamically
        # From num_timesteps in shared memory (already computed in C++)
        'num_state_timesteps': shm_data['num_timesteps'],

        # Source names (generated based on dynamic timesteps)
        'source_name': [f'Kr-88-{i+1}' for i in range(shm_data['num_timesteps'])],

        # Source_1 (true emission source for reference simulation)
        # Format: [decay_constant, DCF, [x,y,z], [emission_series], 0.0, 0.0, 'nuclide']
        'Source_1': [
            6.779608573551890e-05,  # Kr-88 decay constant
            1.02e-13,                # Dose conversion factor
            [10.0, 10.0, 10.0],      # Source location (x, y, z)
            # Emission time series (dynamic based on num_timesteps)
            [1.90387731e+13, 1.90387731e+13, 1.90387731e+12, 1.90387731e+11,
             1.90387731e+4, 1.90387731e+3, 1.90387731e+2, 1.90387731e+1,
             2.26641204e+13, 2.26641204e+13, 2.26641204e+12, 2.26641204e+11,
             2.26641204e+4, 2.26641204e+3, 2.26641204e+2, 2.26641204e+1,
             1.51170139e+13, 1.51170139e+13, 1.51170139e+12, 1.51170139e+11,
             1.51170139e+4, 1.51170139e+3, 1.51170139e+2, 1.51170139e+1],
            0.0e-0,                  # Reserved field
            0.0e-0,                  # Reserved field
            'Kr-88'                  # Nuclide name
        ],

        # Prior_Source_1 (initial guess for inversion)
        # Format: [decay_constant, DCF, [[x,y,z],[std]], [[emission_series],[std]], 'nuclide']
        'Prior_Source_1': [
            6.779608573551890e-05,  # Kr-88 decay constant
            1.02e-13,                # Dose conversion factor
            [[10.0, 10.0, 100.0], [0.1]],  # Location and std
            # Emission series and std (dynamic length)
            [[shm_data['prior_constant']] * shm_data['num_timesteps'], [shm_data['noise_level']]],
            'Kr-88'                  # Nuclide name
        ],

        # Prior source (default constant emission values)
        'prior_source1': [1.0e+14, 1.0e+13, 6.77960857355189e-05],  # Kr-88

        # Boundary
        'real_source1_boundary': [0.0, 1.0e+14],  # Kr-88
    }

    print(f"{Fore.GREEN}✓{Style.RESET_ALL} Configuration loaded:")
    print(f"  Ensemble size      : {Style.BRIGHT}{input_config['sample_ctrl']}{Style.RESET_ALL}")
    print(f"  Iterations         : {Style.BRIGHT}{input_config['iteration']}{Style.RESET_ALL}")
    print(f"  Receptors          : {Style.BRIGHT}{input_data['nreceptor']}{Style.RESET_ALL}")
    print(f"  Sources            : {Style.BRIGHT}{input_data['nsource']}{Style.RESET_ALL}")
    print(f"  GPU devices        : {Style.BRIGHT}{input_config['nGPU']}{Style.RESET_ALL}")

    return input_config, input_data


def print_all_eki_data():
    """
    Debug function to print all EKI configuration and observation data from shared memory.

    Displays:
    1. Full configuration parameters (21 fields)
    2. Constructed input_config dictionary
    3. Constructed input_data dictionary
    4. Observation matrix (if available)
    """
    print("\n" + "="*80)
    print("📊 EKI SHARED MEMORY DATA VERIFICATION")
    print("="*80)

    # 1. Raw shared memory configuration
    print("\n[1] RAW SHARED MEMORY CONFIGURATION (128 bytes)")
    print("-" * 80)
    try:
        shm_config = read_eki_full_config_shm()
        for key, value in shm_config.items():
            if isinstance(value, float):
                print(f"  {key:30s} = {value:.6e}")
            else:
                print(f"  {key:30s} = {value}")
    except Exception as e:
        print(f"  ❌ Failed to read config: {e}")
        return

    # 2. Constructed input_config dictionary
    print("\n[2] INPUT_CONFIG DICTIONARY (Python EKI Parameters)")
    print("-" * 80)
    input_config, input_data = load_config_from_shared_memory()

    print("  EKI Algorithm Settings:")
    print(f"    sample_ctrl (ensemble_size)  : {input_config['sample_ctrl']}")
    print(f"    iteration                    : {input_config['iteration']}")
    print(f"    Optimizer_order              : {input_config['Optimizer_order']}")
    print(f"    perturb_option               : {input_config['perturb_option']}")
    print(f"    Adaptive_EKI                 : {input_config['Adaptive_EKI']}")
    print(f"    Localized_EKI                : {input_config['Localized_EKI']}")
    print(f"    Regularization               : {input_config['Regularization']}")
    print(f"    REnKF_lambda                 : {input_config['REnKF_lambda']}")

    print("\n  GPU Configuration:")
    print(f"    GPU_ForwardPhysicsModel      : {input_config['GPU_ForwardPhysicsModel']}")
    print(f"    GPU_InverseModel             : {input_config['GPU_InverseModel']}")
    print(f"    nGPU                         : {input_config['nGPU']}")

    print("\n  Other Options:")
    print(f"    Elimination                  : {input_config['Elimination']}")
    print(f"    Receptor_Increment           : {input_config['Receptor_Increment']}")

    # 3. Constructed input_data dictionary
    print("\n[3] INPUT_DATA DICTIONARY (Simulation Parameters)")
    print("-" * 80)

    print("  Time Parameters:")
    print(f"    time (days)                  : {input_data['time']}")
    print(f"    time_interval (minutes)      : {input_data['time_interval']}")
    print(f"    inverse_time_interval (hours): {input_data['inverse_time_interval']}")
    print(f"    ave_t                        : {input_data['ave_t']}")

    print("\n  Receptor Parameters:")
    print(f"    nreceptor                    : {input_data['nreceptor']}")
    print(f"    nreceptor_err                : {input_data['nreceptor_err']}")
    print(f"    nreceptor_MDA                : {input_data['nreceptor_MDA']}")
    print(f"    receptor_position (first 3)  : {input_data['receptor_position'][:3]}")

    print("\n  Source Parameters:")
    print(f"    Source_location              : {input_data['Source_location']}")
    print(f"    nsource                      : {input_data['nsource']}")
    print(f"    source_name (first 5)        : {input_data['source_name'][:5]}")
    print(f"    prior_source1                : {input_data['prior_source1']}")
    print(f"    real_source1_boundary        : {input_data['real_source1_boundary']}")

    # 4. Observation matrix (if available)
    print("\n[4] OBSERVATION MATRIX (from shared memory)")
    print("-" * 80)
    try:
        reader = EKIIPCReader()
        ensemble_size, num_receptors, num_timesteps = reader.read_eki_config()
        observations = reader.read_eki_observations()

        print(f"  Shape: {observations.shape} (receptors × timesteps)")
        print(f"  Data type: {observations.dtype}")
        print(f"  Min value: {observations.min():.6e}")
        print(f"  Max value: {observations.max():.6e}")
        print(f"  Mean value: {observations.mean():.6e}")

        print("\n  First 5 timesteps for each receptor:")
        for r in range(min(num_receptors, 3)):
            print(f"    Receptor {r+1}: {observations[r, :5]}")

        print("\n  Last 5 timesteps for each receptor:")
        for r in range(min(num_receptors, 3)):
            print(f"    Receptor {r+1}: {observations[r, -5:]}")

    except Exception as e:
        print(f"  ⚠️  Observation data not yet available: {e}")

    print("\n" + "="*80)
    print("✅ EKI DATA VERIFICATION COMPLETE")
    print("="*80 + "\n")


class EKIConfigManager:
    """
    Manages EKI configuration parameters received from LDM-EKI simulation.
    Stores and provides access to ensemble size, receptor count, and timestep count.
    """
    
    def __init__(self):
        self.ensemble_size = None
        self.num_receptors = None
        self.num_timesteps = None
        self._is_loaded = False
    
    def load_from_shared_memory(self):
        """Load configuration from shared memory."""
        try:
            reader = EKIIPCReader()
            self.ensemble_size, self.num_receptors, self.num_timesteps = reader.read_eki_config()
            self._is_loaded = True
        except Exception as e:
            print(f"\033[91m\033[1m[INPUT ERROR]\033[0m Failed to load EKI configuration from shared memory")
            print()
            print(f"  \033[93mProblem:\033[0m")
            print(f"    Cannot read EKI configuration from /dev/shm/ldm_eki_config.")
            print(f"    Error: {e}")
            print()
            print(f"  \033[96mRequired action:\033[0m")
            print(f"    1. Check that LDM has written configuration to shared memory")
            print(f"    2. Verify /dev/shm permissions (should be rw-rw---- or 0660)")
            print(f"    3. Ensure shared memory size is sufficient")
            print()
            print(f"  \033[92mDebugging steps:\033[0m")
            print(f"    ls -lh /dev/shm/ldm_eki*  # Check files exist")
            print(f"    df -h /dev/shm            # Check available space")
            print(f"    cat /dev/shm/ldm_eki_config | od -t x1  # Inspect raw bytes")
            print()
            print(f"  \033[96mFix in:\033[0m Verify LDM C++ simulation is running and writing config")
            import sys
            sys.exit(1)
    
    def get_ensemble_size(self):
        if not self._is_loaded:
            self.load_from_shared_memory()
        return self.ensemble_size
    
    def get_num_receptors(self):
        if not self._is_loaded:
            self.load_from_shared_memory()
        return self.num_receptors
    
    def get_num_timesteps(self):
        if not self._is_loaded:
            self.load_from_shared_memory()
        return self.num_timesteps
    
    def is_loaded(self):
        return self._is_loaded

# Global EKI configuration manager
eki_config = EKIConfigManager()

desired_gpu_index_cupy = 0

def receive_gamma_dose_matrix(rows, cols):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("127.0.0.1", 8080))

    total_bytes = rows * cols * 4

    data = b""
    while len(data) < total_bytes:
        packet = client_socket.recv(total_bytes - len(data))
        if not packet:
            break
        data += packet

    client_socket.close()

    # Little Endian을 위한 '<' 추가
    gamma_dose_matrix = struct.unpack(f"<{rows * cols}f", data)
    
    # reshape 후 transpose 적용, 그리고 expand_dims로 차원 추가
    reshaped_data = np.expand_dims(np.array(gamma_dose_matrix).reshape((cols, rows)).transpose(), axis=0)
    return reshaped_data

def receive_gamma_dose_matrix_shm_wrapper():
    """
    New shared memory-based function to replace TCP socket communication.
    
    Uses POSIX shared memory for high-performance data transfer from LDM-EKI simulation.
    
    Returns:
        3D numpy array of shape (1, num_receptors, num_timesteps) 
    """
    try:
        gamma_dose_data = receive_gamma_dose_matrix_shm()
        print(f"{Fore.BLUE}[IPC]{Style.RESET_ALL} Received initial observations: {gamma_dose_data.shape}")

        # Exit immediately after displaying the matrix for testing
        # import sys
        # sys.exit(0)

        return gamma_dose_data
    except Exception as e:
        print(f"\033[91m\033[1m[INPUT ERROR]\033[0m Failed to read initial observations from shared memory")
        print()
        print(f"  \033[93mProblem:\033[0m")
        print(f"    Cannot read observation data from /dev/shm/ldm_eki_data.")
        print(f"    Error: {e}")
        print()
        print(f"  \033[96mRequired action:\033[0m")
        print(f"    1. Verify LDM has completed initial simulation")
        print(f"    2. Check /dev/shm/ldm_eki_data exists and is readable")
        print(f"    3. Ensure observation matrix format is correct")
        print()
        print(f"  \033[92mDebugging steps:\033[0m")
        print(f"    ls -lh /dev/shm/ldm_eki_data  # Check file exists")
        print(f"    od -t f4 -N 48 /dev/shm/ldm_eki_data  # Inspect first 12 floats")
        print(f"    grep 'EKI_OBS' logs/ldm_eki_simulation.log  # Check LDM wrote data")
        print()
        print(f"  \033[96mFix in:\033[0m Verify LDM simulation completed successfully")
        import sys
        sys.exit(1)

def send_tmp_states(tmp_states):
    """Legacy TCP socket-based state transmission (deprecated)."""
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("127.0.0.1", 8080))

    data = tmp_states.astype(np.float32).tobytes()
    client_socket.sendall(struct.pack("I", len(data)))
    client_socket.sendall(data)

    print("tmp_states data sent successfully!")
    client_socket.close()

def send_tmp_states_shm(tmp_states):
    """
    Send ensemble states to LDM via POSIX shared memory.

    Replaces the legacy TCP socket communication with high-performance
    shared memory IPC.

    Args:
        tmp_states: 2D numpy array of shape (num_states, num_ensemble)
                   e.g., (24, 100) for 24 timesteps × 100 ensemble members
    """
    from eki_ipc_writer import write_ensemble_to_shm

    num_states, num_ensemble = tmp_states.shape

    print(f"[EKI] Sending ensemble states via shared memory: {num_states}×{num_ensemble}")
    print(f"[EKI] Data range: [{tmp_states.min():.3e}, {tmp_states.max():.3e}]")

    success = write_ensemble_to_shm(tmp_states, num_states, num_ensemble)

    if success:
        print("[EKI] Ensemble states successfully sent to LDM via shared memory")
    else:
        print("[EKI] WARNING: Failed to send ensemble states")
        raise RuntimeError("Failed to write ensemble states to shared memory")

def receive_gamma_dose_matrix_ens(Nens, Nrec):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('127.0.0.1', 12345))
    server_socket.listen(1)
    
    conn, addr = server_socket.accept()
    print(f"Connection from {addr} has been established.")

    # Receive all data at once
    data = b""
    while True:
        packet = conn.recv(4096)
        if not packet:
            break
        data += packet

    conn.close()
    server_socket.close()

    data_str = data.decode()
    lines = data_str.strip().split("\n")

    # Calculate number of state timesteps dynamically from shared memory
    shm_config = read_eki_full_config_shm()
    num_state_timesteps = int(shm_config['time_days'] * 24 / shm_config['inverse_time_interval'])
    h_gamma_dose_3d = np.zeros((Nens, Nrec, num_state_timesteps))
    for line in lines:
        values = line.split(',')
        # print("Values:", values)
        ens = int(values[0])   
        # print("Ens:", ens)  
        t = int(values[1]) - 1    
        # print("Time Index (t):", t)  
        receptors = list(map(float, values[2:]))
        h_gamma_dose_3d[ens, :, t] = receptors  

    return h_gamma_dose_3d

# Forward model interface
class Model(object):
    """
    Forward model interface for LDM-EKI ensemble simulation.

    This class manages the interaction between the Python EKI optimizer and the
    LDM C++/CUDA forward model via POSIX shared memory. It handles:
        - Configuration parameter management
        - Initial observation loading
        - Prior ensemble generation
        - Ensemble state transmission to LDM
        - Ensemble observation reception from LDM

    Attributes
    ----------
    name : str
        Model identifier ('gaussian_puff_model')
    nGPU : int
        Number of GPU devices available
    sample : int
        Number of ensemble members
    nsource : int
        Number of emission sources
    nreceptor : int
        Number of receptor locations
    nstate : int
        Number of state variables (e.g., 24 for 24-hour emission time series)
    obs : ndarray of shape (nreceptor * num_timesteps,)
        Flattened observation vector from LDM initial simulation
    obs_err : ndarray
        Diagonal observation error covariance matrix
    state_init : ndarray of shape (nstate,)
        Prior mean state vector
    state_std : ndarray of shape (nstate,)
        Prior standard deviation vector

    Notes
    -----
    Unlike legacy implementations using Gaussian puff models, this version
    interfaces directly with the LDM Lagrangian particle dispersion model
    running on GPU via shared memory IPC.

    Examples
    --------
    >>> model = Model(input_config, input_data)
    >>> prior_ensemble = model.make_ensemble()  # Generate prior
    >>> observations = model.state_to_ob(ensemble_states)  # Forward model
    """
    def __init__(self, input_config, input_data):
        """
        Initialize forward model interface.

        Parameters
        ----------
        input_config : dict
            Configuration dictionary containing:
            - sample: Ensemble size
            - nGPU: Number of GPU devices
            - (other EKI algorithm parameters)
        input_data : dict
            Input data dictionary containing:
            - nsource: Number of sources
            - nreceptor: Number of receptors
            - Source_1, Prior_Source_1: True and prior source definitions
            - receptor positions, error levels, etc.

        Notes
        -----
        Constructor waits for initial observations from LDM to be available
        in shared memory before completing initialization. This ensures the
        observation vector (self.obs) is populated before EKI loop starts.
        """
        self.name = 'gaussian_puff_model'
        self.nGPU = input_config['nGPU']
        self.input_data = input_data
        self.sample = input_config['sample']
        self.nsource = input_data['nsource']
        self.nreceptor = input_data['nreceptor']
        self.nreceptor_err = input_data['nreceptor_err']
        self.nreceptor_MDA = input_data['nreceptor_MDA']
        if input_data['receptor_position'] == []:
            input_data['receptor_position'] = [list(np.random.randint(low=[input_data['xreceptor_min'], input_data['yreceptor_min'], input_data['zreceptor_min']], high=[input_data['xreceptor_max']+1, input_data['yreceptor_max']+1, input_data['zreceptor_max']+1]).astype(float)) for _ in range(self.nreceptor)]

        # read the real data of sources
        self.real_state_init_list = []
        self.real_decay_list = []
        self.real_source_location_list = []
        self.real_dosecoeff_list = []
        self.total_real_state_list = []

        for s in range(self.nsource):
            actual_source = "Source_{0}".format(s+1)
            self.real_decay_list.append(self.input_data[actual_source][0])
            self.real_dosecoeff_list.append(self.input_data[actual_source][1])
            self.real_source_location_list.append(self.input_data[actual_source][2])
            self.real_state_init_list.append(self.input_data[actual_source][3])
            self.total_real_state_list.append(input_data[actual_source][2] + input_data[actual_source][3])
  
        self.real_state_init = np.array(self.real_state_init_list).reshape(-1)
        self.real_decay = np.array(self.real_decay_list)
        self.real_source_location = np.array(self.real_source_location_list).T
        self.real_dosecoeff = np.array(self.real_dosecoeff_list)
        
        # read the prior data of sources
        self.state_init_list = []
        self.source_location_list = []
        self.state_std_list = []
        self.source_location_std_list = []
        self.decay_list = []
        self.total_state_list = []
        self.total_state_std_list = []
        for s in range(self.nsource):
            source = "Prior_Source_{0}".format(s+1)
            self.source_location_list.append(input_data[source][2][0])
            self.state_init_list.append(input_data[source][3][0])
            self.source_location_std_list.append((np.array(input_data[source][2][0])*input_data[source][2][1]).tolist())
            self.state_std_list.append((np.array(input_data[source][3][0])*input_data[source][3][1]).tolist())
            self.decay_list.append(input_data[source][0])
            self.total_state_list.append(input_data[source][2][0] + input_data[source][3][0])
            self.total_state_std_list.append((np.array(input_data[source][2][0])*input_data[source][2][1]).tolist() + (np.array(input_data[source][3][0])*input_data[source][3][1]).tolist())

        # v1.0: Always use Fixed source location (known position, estimate emissions only)
        self.real_state_init = np.hstack(self.real_state_init_list)
        self.state_init = np.hstack(self.state_init_list)
        self.state_std = np.hstack(self.state_std_list)
        self.nstate_partial = np.array(self.state_init_list).shape[1]
        self.source_location_case = 0  # Fixed source location mode

        self.state_init = np.array(self.state_init).reshape(-1)
        self.state_std = np.array(self.state_std).reshape(-1)

        # Save prior state to logs2/dev
        np.save('/home/jrpark/ekitest2/logs2/dev/prior_state.npy', self.state_init)
        with open('/home/jrpark/ekitest2/logs2/dev/prior_state.txt', 'w') as f:
            f.write(f"Prior State Data\n")
            f.write(f"Shape: {self.state_init.shape}\n")
            f.write(f"Min: {self.state_init.min():.12e}\n")
            f.write(f"Max: {self.state_init.max():.12e}\n")
            f.write(f"Mean: {self.state_init.mean():.12e}\n\n")
            f.write(f"Data ({len(self.state_init)} values):\n")
            for i, val in enumerate(self.state_init):
                f.write(f"{i:4d}: {val:.12e}\n")
        print(f"[Model] Saved prior_state to logs2/dev/")

        self.decay = self.real_decay
        self.nstate = len(self.state_init)
        self.nstate_partial = np.array(self.state_init_list).shape[1]

        # self.gpuff = Gpuff_Ensemble.PuffSimulation(self.input_data)
        # self.gpuff.create_context()
        # input_data_ensemble_init = []
        # input_data_ensemble_init.append(deepcopy(self.input_data))

        # IMPORTANT: Read initial observations in __init__ to match reference code behavior
        # Unlike reference code which can read from files immediately, we need to wait
        # for LDM to write the data to shared memory first

        print(f"[Model.__init__] Waiting for initial observations from LDM...")

        # Wait for initial observations to be available in shared memory
        import time
        import os
        max_wait = 30  # Maximum wait time in seconds
        wait_interval = 0.1  # Check every 100ms
        start_time = time.time()

        obs_config_path = "/dev/shm/ldm_eki_config"
        obs_data_path = "/dev/shm/ldm_eki_data"

        # Poll for observation files
        observations_ready = False
        while (time.time() - start_time) < max_wait:
            if os.path.exists(obs_config_path) and os.path.exists(obs_data_path):
                try:
                    # Check file sizes to ensure data is written
                    config_size = os.path.getsize(obs_config_path)
                    data_size = os.path.getsize(obs_data_path)
                    if config_size > 0 and data_size > 0:
                        print(f"[Model.__init__] Initial observations detected after {time.time() - start_time:.2f}s")
                        observations_ready = True
                        time.sleep(0.05)  # Small delay to ensure write is complete
                        break
                except:
                    pass  # File may be in process of being written
            time.sleep(wait_interval)

        if not observations_ready:
            print(f"\033[91m\033[1m[INPUT ERROR]\033[0m Initial observations not available after {max_wait}s timeout")
            print()
            print(f"  \033[93mProblem:\033[0m")
            print(f"    LDM has not written initial observations to shared memory.")
            print(f"    Files expected:")
            print(f"      - /dev/shm/ldm_eki_config (configuration)")
            print(f"      - /dev/shm/ldm_eki_data (observation matrix)")
            print()
            print(f"  \033[96mRequired action:\033[0m")
            print(f"    1. Check that LDM process is running")
            print(f"    2. Verify LDM has completed initial simulation")
            print(f"    3. Check logs/ldm_eki_simulation.log for errors")
            print()
            print(f"  \033[92mDebugging steps:\033[0m")
            print(f"    ps aux | grep ldm-eki     # Check LDM process")
            print(f"    ls -lh /dev/shm/ldm_eki*  # Check shared memory files")
            print(f"    tail -f logs/ldm_eki_simulation.log  # Monitor LDM output")
            print()
            print(f"  \033[96mFix in:\033[0m Ensure LDM is running and writing observations")
            import sys
            sys.exit(1)
        else:
            # Read initial observations from shared memory (same as reference code reads from socket)
            print(f"[Model.__init__] Reading initial observations from shared memory...")
            gamma_dose_data = receive_gamma_dose_matrix_shm_wrapper()
            print(f"[Model.__init__] Received gamma dose data shape: {np.array(gamma_dose_data).shape}")
            print(f"[Model.__init__] Received gamma dose data stats - min: {gamma_dose_data.min():.3e}, max: {gamma_dose_data.max():.3e}, mean: {gamma_dose_data.mean():.3e}")

            # Store observations (same as reference: line 237)
            self.obs = np.array(gamma_dose_data[0]).reshape(-1)

            # Save initial observation to logs2/dev
            np.save('/home/jrpark/ekitest2/logs2/dev/initial_observation.npy', self.obs)
            # Also save as text for easy comparison
            with open('/home/jrpark/ekitest2/logs2/dev/initial_observation.txt', 'w') as f:
                f.write(f"Initial Observation Data\n")
                f.write(f"Shape: {self.obs.shape}\n")
                f.write(f"Min: {self.obs.min():.12e}\n")
                f.write(f"Max: {self.obs.max():.12e}\n")
                f.write(f"Mean: {self.obs.mean():.12e}\n")
                f.write(f"Sum: {self.obs.sum():.12e}\n\n")
                f.write(f"Data (flattened, {len(self.obs)} values):\n")
                for i, val in enumerate(self.obs):
                    f.write(f"{i:4d}: {val:.12e}\n")
            print(f"[Model.__init__] Saved initial_observation to logs2/dev/")

        # Initialize error matrices (same as reference: lines 241-242)
        self.obs_err = np.diag((np.floor(self.obs * 0) + np.ones([len(self.obs)])*self.nreceptor_err))  # nreceptor_err (percentage), needed to square later
        self.obs_MDA = np.diag((np.floor(self.obs * 0) + np.ones([len(self.obs)])*self.nreceptor_MDA))   # nreceptor_err (percentage), needed to square later

        # read the real data of sources's boundary for PSO
        self.lowerbounds_list = []
        self.upperbounds_list = []
        for s in range(self.nsource):
            real_source = "real_source{0}_boundary".format(s+1)
            for r in range(12):
                self.lowerbounds_list.append(input_data[real_source][0])
                self.upperbounds_list.append(input_data[real_source][1])
        self.bounds = np.array([self.lowerbounds_list, self.upperbounds_list]).T

    def __str__(self):
        """Return model name string."""
        return self.name

    def make_ensemble(self):
        """
        Generate prior ensemble from Gaussian distribution.

        Returns
        -------
        state : ndarray of shape (nstate, sample)
            Prior ensemble matrix where each column is an ensemble member.
            Shape: (num_states, num_ensemble), e.g., (24, 100) for 24 timesteps
            and 100 ensemble members.

        Notes
        -----
        - Uses absolute value to prevent negative emission rates
        - Random seed should be set externally for reproducibility
        - State variables are sampled independently (diagonal prior covariance)
        - Saved to logs2/dev/prior_ensemble.npy for debugging

        Examples
        --------
        >>> prior = model.make_ensemble()
        >>> prior.shape
        (24, 100)  # 24 timesteps, 100 ensemble members
        """
        state = np.empty([self.nstate, self.sample])
        for i in range(self.nstate):
            # Use abs() to prevent negative initial values (same as reference)
            state[i, :] = np.abs(np.random.normal(self.state_init[i], self.state_std[i], self.sample))

        # Save prior ensemble to logs2/dev
        np.save('/home/jrpark/ekitest2/logs2/dev/prior_ensemble.npy', state)
        with open('/home/jrpark/ekitest2/logs2/dev/prior_ensemble.txt', 'w') as f:
            f.write(f"Prior Ensemble Data\n")
            f.write(f"Shape: {state.shape} (num_states x num_ensemble)\n")
            f.write(f"Min: {state.min():.12e}\n")
            f.write(f"Max: {state.max():.12e}\n")
            f.write(f"Mean: {state.mean():.12e}\n")
            f.write(f"Std: {state.std():.12e}\n\n")
            f.write(f"First 5 timesteps, all ensembles:\n")
            for t in range(min(5, state.shape[0])):
                f.write(f"\nTimestep {t}:\n")
                for e in range(min(10, state.shape[1])):
                    f.write(f"  Ens{e:3d}: {state[t, e]:.12e}\n")
        print(f"[Model] Saved prior_ensemble to logs2/dev/")
        print(f"[Model] Prior ensemble shape: {state.shape}, range: [{state.min():.3e}, {state.max():.3e}]")

        return state
    

    def state_to_ob(self, state):
        """
        Forward model: Map ensemble states to ensemble observations via LDM.

        This is the core forward model operator that:
        1. Sends ensemble states to LDM via shared memory
        2. Waits for LDM to complete ensemble simulations
        3. Receives ensemble observations from LDM via shared memory
        4. Reshapes observations to match EKI format

        Parameters
        ----------
        state : ndarray of shape (nstate, sample)
            Ensemble state matrix (column-major format).
            Each column is one ensemble member.
            Shape: (num_states, num_ensemble), e.g., (24, 100)

        Returns
        -------
        model_obs : ndarray of shape (nreceptor * num_timesteps, sample)
            Ensemble observation matrix where each column contains observations
            for one ensemble member. Format: [R0_T0...T23, R1_T0...T23, ...]

        Notes
        -----
        **IPC Communication Flow:**
        1. Python writes config: (num_states, num_ensemble, iteration_id)
        2. Python writes states: Row-major flatten of (nstate, sample)
        3. LDM detects new data via iteration_id
        4. LDM runs ensemble simulations (N members in parallel)
        5. LDM writes observations: [ens][timestep][receptor]
        6. Python reads observations and transposes to match EKI format

        **Array Layout Conversions:**

        The IPC transfer requires careful attention to memory layouts due to
        NumPy (column-major by default) vs C++ (row-major) differences.

        *State Transmission (Python → C++):*
        - Python format: (num_states, num_ensemble) = (24, 100)
          Column-major: state[timestep, ensemble]
        - Flatten with order='C': Row-major sequential layout
        - C++ reads as: states[ensemble][timestep] via row-major indexing

        *Observation Reception (C++ → Python):*
        - C++ format: [ensemble][timestep][receptor] = [100][24][3]
          Memory layout: Ens0_T0_R0, Ens0_T0_R1, Ens0_T0_R2, Ens0_T1_R0, ...
        - Python reads as: (ensemble, timestep, receptor) with order='C'
        - Transpose to: (ensemble, receptor, timestep)
        - Flatten each ensemble: receptor-major [R0_T0...T23, R1_T0...T23, R2_T0...T23]
        - Final Python format: (nobs, num_ensemble) = (72, 100)

        **Memory Layout Example:**

        State transmission (24 timesteps, 3 ensembles):
            Python: states = [[e1_t1, e2_t1, e3_t1],    # shape (24, 3)
                              [e1_t2, e2_t2, e3_t2],
                              ...
                              [e1_t24, e2_t24, e3_t24]]

            Flatten('C'): [e1_t1, e2_t1, e3_t1, e1_t2, e2_t2, e3_t2, ...]
                          (row-major, state-major within ensemble)

            C++ reads: states[0][0]=e1_t1, states[0][1]=e1_t2, ...
                       states[1][0]=e2_t1, states[1][1]=e2_t2, ...

        Observation reception (3 receptors, 24 timesteps, 2 ensembles):
            C++ writes: [Ens0: T0_R0, T0_R1, T0_R2, T1_R0, T1_R1, T1_R2, ...]
                        [Ens1: T0_R0, T0_R1, T0_R2, T1_R0, T1_R1, T1_R2, ...]

            Python reads: obs[0, 0, 0]=Ens0_T0_R0, obs[0, 0, 1]=Ens0_T0_R1, ...
                          shape (2, 24, 3)

            Transpose: obs[:, :, :] → obs[:, receptor, timestep]
                       shape (2, 3, 24)

            Flatten each ensemble: [R0_T0, R0_T1, ..., R0_T23,
                                   R1_T0, R1_T1, ..., R1_T23,
                                   R2_T0, R2_T1, ..., R2_T23]

            Final: shape (72, 2) - each column is one ensemble's observations

        **Performance:**
        - Typical wait time: 5-30 seconds for 100 ensemble members
        - Timeout after 120 seconds with error

        Examples
        --------
        >>> state = model.make_ensemble()  # (24, 100)
        >>> observations = model.state_to_ob(state)  # (72, 100) for 3 receptors
        """

        model_obs_list = []
        tmp_states = state.copy()
        #np.set_printoptions(threshold=np.inf)
        print(f"{Fore.MAGENTA}[ENSEMBLE]{Style.RESET_ALL} Forward model: {tmp_states.shape[0]} states × {tmp_states.shape[1]} members")

        # IMPORTANT: Delete previous observation files to avoid reading stale data
        import os
        ensemble_obs_config_path = "/dev/shm/ldm_eki_ensemble_obs_config"
        ensemble_obs_data_path = "/dev/shm/ldm_eki_ensemble_obs_data"

        # Clean previous observation files
        for path in [ensemble_obs_config_path, ensemble_obs_data_path]:
            if os.path.exists(path):
                os.remove(path)

        # NOTE: Don't delete ensemble state files - just overwrite them
        # C++ waitForEnsembleData() uses iteration ID to detect new data
        # Deleting files causes race condition where C++ may miss the new write

        # MODIFIED: Use shared memory instead of TCP socket
        # IMPORTANT: Track iteration count to help LDM detect new data
        if not hasattr(self, '_iteration_counter'):
            self._iteration_counter = 0
        self._iteration_counter += 1

        # Save ensemble states being sent to logs2/dev
        np.save(f'/home/jrpark/ekitest2/logs2/dev/iter{self._iteration_counter:03d}_ensemble_states_sent.npy', tmp_states)
        with open(f'/home/jrpark/ekitest2/logs2/dev/iter{self._iteration_counter:03d}_ensemble_states_sent.txt', 'w') as f:
            f.write(f"Iteration {self._iteration_counter} - Ensemble States Sent (Python → C++)\n")
            f.write(f"Shape: {tmp_states.shape} (num_timesteps x num_ensemble)\n")
            f.write(f"Min: {tmp_states.min():.12e}\n")
            f.write(f"Max: {tmp_states.max():.12e}\n")
            f.write(f"Mean: {tmp_states.mean():.12e}\n")
            f.write(f"Std: {tmp_states.std():.12e}\n\n")
            f.write(f"First 5 timesteps, first 10 ensembles:\n")
            for t in range(min(5, tmp_states.shape[0])):
                f.write(f"\nTimestep {t}:\n")
                for e in range(min(10, tmp_states.shape[1])):
                    f.write(f"  Ens{e:3d}: {tmp_states[t, e]:.12e}\n")
        print(f"[EKI] Saved iteration {self._iteration_counter} ensemble states to logs2/dev/")

        # Pass iteration counter as timestep_id to help detect fresh data
        from eki_ipc_writer import EKIIPCWriter
        writer = EKIIPCWriter()
        writer.write_ensemble_config(tmp_states.shape[0], tmp_states.shape[1], self._iteration_counter)
        writer.write_ensemble_states(tmp_states, tmp_states.shape[0], tmp_states.shape[1])
        print(f"{Fore.BLUE}[IPC]{Style.RESET_ALL} Sent ensemble states (iteration {self._iteration_counter})")

        # Wait for LDM to complete ensemble simulation and write observations
        print(f"{Fore.CYAN}[SYSTEM]{Style.RESET_ALL} Waiting for LDM simulation...", end='', flush=True)
        import time
        import os

        # Configuration
        max_wait_time = 120  # Maximum wait time in seconds
        poll_interval = 0.5  # Check every 0.5 seconds
        ensemble_obs_config_path = "/dev/shm/ldm_eki_ensemble_obs_config"
        ensemble_obs_data_path = "/dev/shm/ldm_eki_ensemble_obs_data"

        # Poll for ensemble observation files
        start_time = time.time()
        files_ready = False

        while (time.time() - start_time) < max_wait_time:
            # Check if both config and data files exist
            if os.path.exists(ensemble_obs_config_path) and os.path.exists(ensemble_obs_data_path):
                # Additional check: make sure files have non-zero size
                try:
                    config_size = os.path.getsize(ensemble_obs_config_path)
                    data_size = os.path.getsize(ensemble_obs_data_path)
                    if config_size > 0 and data_size > 0:
                        print(f" {Fore.GREEN}✓{Style.RESET_ALL} ({time.time() - start_time:.1f}s)")
                        files_ready = True
                        # Small delay to ensure write is complete
                        time.sleep(0.1)
                        break
                except OSError:
                    pass  # File may have been deleted between exists() and getsize()

            time.sleep(poll_interval)

        if not files_ready:
            print(f" {Fore.RED}✗{Style.RESET_ALL}")
            raise TimeoutError(f"{Fore.RED}{Style.BRIGHT}[ERROR]{Style.RESET_ALL} Timeout waiting for LDM ({max_wait_time}s)\n"
                             f"  → Check if LDM process is running\n"
                             f"  → Check logs/ldm_eki_simulation.log for errors")

        # Receive ensemble observations from LDM via shared memory (pass iteration for logging)
        from eki_ipc_reader import receive_ensemble_observations_shm
        tmp_results = receive_ensemble_observations_shm(self._iteration_counter)
        # After C++ fix, shape is now: [num_ensemble, num_timesteps, num_receptors]
        # This matches reference implementation: timestep-major within each ensemble
        print(f"{Fore.BLUE}[IPC]{Style.RESET_ALL} Received ensemble observations: {tmp_results.shape}")

        # Save all iteration ensemble observations to logs2/dev
        # Convert to reference format: (ensemble, receptors, timesteps)
        # tmp_results is currently (ensemble, timesteps, receptors)
        tmp_results_transposed = np.transpose(tmp_results, (0, 2, 1))
        np.save(f'/home/jrpark/ekitest2/logs2/dev/iter{self._iteration_counter:03d}_ensemble_observations_received.npy', tmp_results_transposed)
        with open(f'/home/jrpark/ekitest2/logs2/dev/iter{self._iteration_counter:03d}_ensemble_observations_received.txt', 'w') as f:
            f.write(f"Iteration {self._iteration_counter} - Ensemble Observations Received (C++ → Python)\n")
            f.write(f"Shape: {tmp_results_transposed.shape} (num_ensemble x num_receptors x num_timesteps)\n")
            f.write(f"Min: {tmp_results_transposed.min():.12e}\n")
            f.write(f"Max: {tmp_results_transposed.max():.12e}\n")
            f.write(f"Mean: {tmp_results_transposed.mean():.12e}\n")
            f.write(f"Std: {tmp_results_transposed.std():.12e}\n\n")
            f.write(f"First ensemble, all receptors, first 5 timesteps:\n")
            for r in range(tmp_results_transposed.shape[1]):
                f.write(f"\nReceptor {r}:\n")
                for t in range(min(5, tmp_results_transposed.shape[2])):
                    f.write(f"  T{t:3d}: {tmp_results_transposed[0, r, t]:.12e}\n")
        print(f"[Model] Saved iteration {self._iteration_counter} ensemble observations to logs2/dev/")

        # Reshape for EKI: Need transpose to match reference final order
        # tmp_results is [ensemble, timestep, receptor] from C++
        # EKI expects [R0_T0...T23, R1_T0...T23, R2_T0...T23]
        for ens in range(tmp_states.shape[1]):
            # tmp_results[ens] is shape (num_timesteps, num_receptors) = (24, 3)
            # Transpose to (num_receptors, num_timesteps) = (3, 24) then flatten
            ensemble_obs = tmp_results[ens].T  # Now (num_receptors, num_timesteps)
            model_obs_list.append(np.asarray(ensemble_obs).reshape(-1))

        del tmp_results

        # gpuff.destroy_context()
        # del gpuff
        model_obs = np.asarray(model_obs_list).T

        # print(np.array(model_obs).shape)
        # print(model_obs)
        return model_obs

    def read_initial_observations(self):
        """
        Read initial observations from LDM (DEPRECATED).

        Notes
        -----
        This function is deprecated. Initial observations are now read in __init__()
        to match reference code behavior. Kept for backward compatibility only.

        The constructor (__init__) now waits for initial observations from LDM
        before completing initialization, ensuring self.obs is populated before
        the EKI loop starts.
        """
        print("[Model] Reading initial observations from LDM...")

        # DEBUG: Check what files exist in shared memory
        import os
        shm_files = [f for f in os.listdir('/dev/shm') if 'ldm_eki' in f]
        print(f"[Model DEBUG] Shared memory files before reading: {shm_files}")

        # Use shared memory to receive initial observation matrix from LDM
        gamma_dose_data = receive_gamma_dose_matrix_shm_wrapper()
        print(f"[Model] Received initial observations shape: {gamma_dose_data.shape}")
        print(f"[Model] Initial observations statistics - min: {gamma_dose_data.min():.3e}, max: {gamma_dose_data.max():.3e}, mean: {gamma_dose_data.mean():.3e}")

        # DEBUG: Check if this looks like observation data or ensemble state data
        # Get expected dimensions from config
        expected_receptors = eki_config.get_num_receptors()
        expected_timesteps = eki_config.get_num_timesteps()

        if gamma_dose_data.shape[0] == 1 and gamma_dose_data.shape[1] == expected_receptors and gamma_dose_data.shape[2] == expected_timesteps:
            print(f"[Model] ✅ This looks like observation data (1 x {expected_receptors} receptors x {expected_timesteps} timesteps)")
        else:
            print(f"[Model] ⚠️ WARNING: Unexpected shape! Expected (1, {expected_receptors}, {expected_timesteps}) for observations")

        # Store as flattened observation vector
        self.obs = np.array(gamma_dose_data[0]).reshape(-1)
        print(f"[Model] Set obs array with size {len(self.obs)}")

        # # Save initial observation to log
        np.save('/home/jrpark/ekitest2/logs/development/initial_observation.npy', self.obs)
        print(f"[Model] Saved initial_observation to logs/development/initial_observation.npy")

        # Recalculate error matrices with actual observation values
        self.obs_err = np.diag((np.floor(self.obs * 0) + np.ones([len(self.obs)])*self.nreceptor_err))
        self.obs_MDA = np.diag((np.floor(self.obs * 0) + np.ones([len(self.obs)])*self.nreceptor_MDA))

    def get_ob(self, time):
        """
        Get observation vector and observation error covariance.

        Parameters
        ----------
        time : int
            Time index (currently unused in EKI, kept for interface compatibility)

        Returns
        -------
        obs : ndarray of shape (nreceptor * num_timesteps,)
            Flattened observation vector from initial LDM simulation
        obs_err : ndarray of shape (nreceptor * num_timesteps, nreceptor * num_timesteps)
            Diagonal observation error covariance matrix

        Notes
        -----
        Observation error is computed as:
            obs_err[i,i] = (obs_rel_std * obs[i] + obs_abs_std)^2

        Where:
            - obs_rel_std = nreceptor_err (relative error percentage)
            - obs_abs_std = nreceptor_MDA (absolute error floor)

        This error model combines relative and absolute uncertainties to avoid
        zero variance when observations are near zero.
        """
        self.obs_err = (self.obs * self.obs_err + self.obs_MDA)**2   # [obs_rel_std(rate) * true_obs + obs_abs_std]**2
        return self.obs, self.obs_err

    def predict(self, state, time):
        """
        Predict next state (identity forecast model).

        Parameters
        ----------
        state : ndarray of shape (nstate, sample)
            Current ensemble state
        time : int
            Time index (unused)

        Returns
        -------
        state : ndarray of shape (nstate, sample)
            Predicted state (zeros for static inversion)

        Notes
        -----
        For static source term estimation, the forecast model is identity
        (no temporal evolution). This method returns zeros and is typically
        not used in the EKI loop.
        """
        state = np.zeros([self.nstate, self.sample])
        return state

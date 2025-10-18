"""
Shared Memory Configuration Management for LDM-EKI

This module handles all shared memory-based configuration and data transfer
between the C++/CUDA LDM simulation and the Python EKI inverse model.

Functions:
    - load_config_from_shared_memory(): Load full configuration from shared memory
    - receive_gamma_dose_matrix_shm_wrapper(): Read initial observations
    - send_tmp_states_shm(): Send ensemble states to LDM

Classes:
    - EKIConfigManager: Configuration manager for shared memory access
"""

import numpy as np
from eki_ipc_reader import receive_gamma_dose_matrix_shm, EKIIPCReader, read_eki_full_config_shm, read_true_emissions_shm
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
    Load configuration from shared memory.

    Reads the full EKI configuration from shared memory segments written by
    the C++ LDM process, and constructs input_config and input_data dictionaries
    compatible with the EKI optimizer.

    Returns:
        tuple: (input_config, input_data) dictionaries containing all parameters
               needed for EKI optimization
    """
    print(f"{Fore.MAGENTA}[ENSEMBLE]{Style.RESET_ALL} Loading configuration from shared memory...")

    # Read full configuration from shared memory
    shm_data = read_eki_full_config_shm()

    # Read true emissions from separate shared memory segment
    true_emissions = read_true_emissions_shm()
    print(f"{Fore.CYAN}[IPC]{Style.RESET_ALL} True emissions loaded: {len(true_emissions)} timesteps")

    memory_doctor_value = shm_data['memory_doctor'].strip()

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

        # HARD-CODED TO 'Off' FOR v1.0 DEPLOYMENT
        # The LOCALIZED option is disabled for this release due to concerns about
        # physical correctness. This is the only allowed hard-coding exception.
        # Future releases will re-enable this after additional validation.
        'Localized_EKI': 'Off',  # Always Off, ignoring shm_data['localized_eki']

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

    # TODO: Extend EKIConfigFull in C++ to include these arrays:
    # - true_emissions[num_timesteps]
    # - receptor_positions[num_receptors][3]  (lat, lon, alt)
    # - source_location[3]  (x, y, z)
    # - decay_constant, dose_conversion_factor
    # - nuclide_name
    # - emission_boundary_min, emission_boundary_max

    # Construct input_data dictionary
    input_data = {
        # Time parameters (from shared memory)
        'time': shm_data['time_interval'] / 60.0,  # Convert minutes to hours
        'time_interval': shm_data['time_interval'],  # EKI time interval (minutes)
        'inverse_time_interval': shm_data['time_interval'] / 60.0,  # Time interval in hours

        # Receptor parameters (from shared memory)
        'nreceptor': shm_data['num_receptors'],

        # TODO: Read from shared memory when C++ sends receptor_positions
        # For now, using placeholder - LDM uses lat/lon from eki.conf, not these XYZ coordinates
        'receptor_position': [[1000.0 * (i+1), 1000.0 * (i+1), 1.0] for i in range(shm_data['num_receptors'])],

        'nreceptor_err': 0.0,  # No additional measurement error (noise in observations)
        'nreceptor_MDA': 0.0,  # No MDA inflation

        # Source parameters (v1.0: Fixed location mode)
        'Source_location': 'Fixed',  # Always use known source position
        'nsource': 1,                # Always single source

        # Number of state timesteps (from shared memory)
        'num_state_timesteps': shm_data['num_timesteps'],

        # Source names (generated dynamically)
        'source_name': [f'Kr-88-{i+1}' for i in range(shm_data['num_timesteps'])],

        # Source_1 (true emission source for reference simulation)
        # Format: [decay_constant, DCF, [x,y,z], [emission_series], 0.0, 0.0, 'nuclide']
        # Values now read from shared memory (no hardcoded values)
        'Source_1': [
            shm_data['decay_constant'],  # Decay constant λ [s⁻¹] from nuclides.conf
            1.02e-13,                    # Dose conversion factor (unused by LDM, kept for compatibility)
            [10.0, 10.0, 10.0],          # Source location (unused by LDM - uses source.conf)
            true_emissions.tolist(),     # True emission time series from eki.conf
            0.0e-0,                      # Reserved field
            0.0e-0,                      # Reserved field
            'Kr-88'                      # Nuclide name (fixed to Kr-88 for v1.0)
        ],

        # Prior_Source_1 (initial guess for inversion)
        # Format: [decay_constant, DCF, [[x,y,z],[std]], [[emission_series],[std]], 'nuclide']
        # Values now read from shared memory (no hardcoded values)
        'Prior_Source_1': [
            shm_data['decay_constant'],  # Decay constant λ [s⁻¹] from nuclides.conf
            1.02e-13,                    # Dose conversion factor (unused by LDM, kept for compatibility)
            [[10.0, 10.0, 100.0], [0.1]],  # Location and std (unused by LDM - uses source.conf)
            # Prior emission: constant value with noise
            [[shm_data['prior_constant']] * shm_data['num_timesteps'], [shm_data['noise_level']]],
            'Kr-88'                      # Nuclide name (fixed to Kr-88 for v1.0)
        ],

        # Prior source bounds (now read from shared memory)
        'prior_source1': [1.0e+14, 1.0e+13, shm_data['decay_constant']],

        # Emission boundary for optimization (kept as constants for v1.0)
        'real_source1_boundary': [0.0, 1.0e+14],
    }

    print(f"{Fore.GREEN}✓{Style.RESET_ALL} Configuration loaded:")
    print(f"  Ensemble size      : {Style.BRIGHT}{input_config['sample_ctrl']}{Style.RESET_ALL}")
    print(f"  Iterations         : {Style.BRIGHT}{input_config['iteration']}{Style.RESET_ALL}")
    print(f"  Receptors          : {Style.BRIGHT}{input_data['nreceptor']}{Style.RESET_ALL}")
    print(f"  Sources            : {Style.BRIGHT}{input_data['nsource']}{Style.RESET_ALL}")
    print(f"  GPU devices        : {Style.BRIGHT}{input_config['nGPU']}{Style.RESET_ALL}")

    return input_config, input_data


class EKIConfigManager:
    """
    Configuration manager for shared memory access.

    Provides lazy loading and caching of configuration parameters from
    shared memory. Used globally to avoid repeated shared memory reads.
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


def receive_gamma_dose_matrix_shm_wrapper():
    """
    Read initial observations from shared memory.

    Wrapper function for receiving the initial gamma dose observation matrix
    from the C++ LDM simulation via POSIX shared memory. This replaces the
    legacy TCP socket communication.

    Returns:
        numpy.ndarray: 3D array of shape (1, num_receptors, num_timesteps)
                      containing gamma dose observations

    Raises:
        SystemExit: If reading from shared memory fails
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


def send_tmp_states_shm(tmp_states):
    """
    Send ensemble states to LDM via shared memory.

    Replaces the legacy TCP socket communication with high-performance
    POSIX shared memory IPC.

    Args:
        tmp_states (numpy.ndarray): 2D array of shape (num_states, num_ensemble)
                                    e.g., (24, 100) for 24 timesteps × 100 ensemble members

    Raises:
        RuntimeError: If writing to shared memory fails
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

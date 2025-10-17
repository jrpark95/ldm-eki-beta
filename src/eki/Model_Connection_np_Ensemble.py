import numpy as np
from copy import deepcopy

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

desired_gpu_index_cupy = 0

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

# def receive_gamma_dose_matrix_ens(Nens, Nrec):
#     server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     server_socket.bind(('127.0.0.1', 12345))
#     server_socket.listen(1)
    
#     conn, addr = server_socket.accept()
#     print(f"Connection from {addr} has been established.")

#     # Receive all data at once
#     data = b""
#     while True:
#         packet = conn.recv(4096)
#         if not packet:
#             break
#         data += packet

#     conn.close()
#     server_socket.close()

#     data_str = data.decode()
#     lines = data_str.strip().split("\n")

#     # Calculate number of state timesteps dynamically from shared memory
#     shm_config = read_eki_full_config_shm()
#     num_state_timesteps = int(shm_config['time_days'] * 24 / shm_config['inverse_time_interval'])
#     h_gamma_dose_3d = np.zeros((Nens, Nrec, num_state_timesteps))
#     for line in lines:
#         values = line.split(',')
#         # print("Values:", values)
#         ens = int(values[0])   
#         # print("Ens:", ens)  
#         t = int(values[1]) - 1    
#         # print("Time Index (t):", t)  
#         receptors = list(map(float, values[2:]))
#         h_gamma_dose_3d[ens, :, t] = receptors  

#     return h_gamma_dose_3d

# Forward model interface
class Model(object):
    def __init__(self, input_config, input_data):
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

        # # Save prior state to logs2/dev
        # np.save('/home/jrpark/ekitest2/logs2/dev/prior_state.npy', self.state_init)
        # with open('/home/jrpark/ekitest2/logs2/dev/prior_state.txt', 'w') as f:
        #     f.write(f"Prior State Data\n")
        #     f.write(f"Shape: {self.state_init.shape}\n")
        #     f.write(f"Min: {self.state_init.min():.12e}\n")
        #     f.write(f"Max: {self.state_init.max():.12e}\n")
        #     f.write(f"Mean: {self.state_init.mean():.12e}\n\n")
        #     f.write(f"Data ({len(self.state_init)} values):\n")
        #     for i, val in enumerate(self.state_init):
        #         f.write(f"{i:4d}: {val:.12e}\n")
        # print(f"[Model] Saved prior_state to logs2/dev/")

        self.decay = self.real_decay
        self.nstate = len(self.state_init)
        self.nstate_partial = np.array(self.state_init_list).shape[1]

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

            # # Save initial observation to logs2/dev
            # np.save('/home/jrpark/ekitest2/logs2/dev/initial_observation.npy', self.obs)
            # # Also save as text for easy comparison
            # with open('/home/jrpark/ekitest2/logs2/dev/initial_observation.txt', 'w') as f:
            #     f.write(f"Initial Observation Data\n")
            #     f.write(f"Shape: {self.obs.shape}\n")
            #     f.write(f"Min: {self.obs.min():.12e}\n")
            #     f.write(f"Max: {self.obs.max():.12e}\n")
            #     f.write(f"Mean: {self.obs.mean():.12e}\n")
            #     f.write(f"Sum: {self.obs.sum():.12e}\n\n")
            #     f.write(f"Data (flattened, {len(self.obs)} values):\n")
            #     for i, val in enumerate(self.obs):
            #         f.write(f"{i:4d}: {val:.12e}\n")
            # print(f"[Model.__init__] Saved initial_observation to logs2/dev/")

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
        return self.name

    def make_ensemble(self):
        state = np.empty([self.nstate, self.sample])
        for i in range(self.nstate):
            # Use abs() to prevent negative initial values (same as reference)
            state[i, :] = np.abs(np.random.normal(self.state_init[i], self.state_std[i], self.sample))

        # # Save prior ensemble to logs2/dev
        # np.save('/home/jrpark/ekitest2/logs2/dev/prior_ensemble.npy', state)
        # with open('/home/jrpark/ekitest2/logs2/dev/prior_ensemble.txt', 'w') as f:
        #     f.write(f"Prior Ensemble Data\n")
        #     f.write(f"Shape: {state.shape} (num_states x num_ensemble)\n")
        #     f.write(f"Min: {state.min():.12e}\n")
        #     f.write(f"Max: {state.max():.12e}\n")
        #     f.write(f"Mean: {state.mean():.12e}\n")
        #     f.write(f"Std: {state.std():.12e}\n\n")
        #     f.write(f"First 5 timesteps, all ensembles:\n")
        #     for t in range(min(5, state.shape[0])):
        #         f.write(f"\nTimestep {t}:\n")
        #         for e in range(min(10, state.shape[1])):
        #             f.write(f"  Ens{e:3d}: {state[t, e]:.12e}\n")
        # print(f"[Model] Saved prior_ensemble to logs2/dev/")
        # print(f"[Model] Prior ensemble shape: {state.shape}, range: [{state.min():.3e}, {state.max():.3e}]")

        return state
    

    def state_to_ob(self, state):

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

        if not hasattr(self, '_iteration_counter'):
            self._iteration_counter = 0
        self._iteration_counter += 1

        # # Save ensemble states being sent to logs2/dev
        # np.save(f'/home/jrpark/ekitest2/logs2/dev/iter{self._iteration_counter:03d}_ensemble_states_sent.npy', tmp_states)
        # with open(f'/home/jrpark/ekitest2/logs2/dev/iter{self._iteration_counter:03d}_ensemble_states_sent.txt', 'w') as f:
        #     f.write(f"Iteration {self._iteration_counter} - Ensemble States Sent (Python → C++)\n")
        #     f.write(f"Shape: {tmp_states.shape} (num_timesteps x num_ensemble)\n")
        #     f.write(f"Min: {tmp_states.min():.12e}\n")
        #     f.write(f"Max: {tmp_states.max():.12e}\n")
        #     f.write(f"Mean: {tmp_states.mean():.12e}\n")
        #     f.write(f"Std: {tmp_states.std():.12e}\n\n")
        #     f.write(f"First 5 timesteps, first 10 ensembles:\n")
        #     for t in range(min(5, tmp_states.shape[0])):
        #         f.write(f"\nTimestep {t}:\n")
        #         for e in range(min(10, tmp_states.shape[1])):
        #             f.write(f"  Ens{e:3d}: {tmp_states[t, e]:.12e}\n")
        # print(f"[EKI] Saved iteration {self._iteration_counter} ensemble states to logs2/dev/")

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
        # tmp_results_transposed = np.transpose(tmp_results, (0, 2, 1))
        # np.save(f'/home/jrpark/ekitest2/logs2/dev/iter{self._iteration_counter:03d}_ensemble_observations_received.npy', tmp_results_transposed)
        # with open(f'/home/jrpark/ekitest2/logs2/dev/iter{self._iteration_counter:03d}_ensemble_observations_received.txt', 'w') as f:
        #     f.write(f"Iteration {self._iteration_counter} - Ensemble Observations Received (C++ → Python)\n")
        #     f.write(f"Shape: {tmp_results_transposed.shape} (num_ensemble x num_receptors x num_timesteps)\n")
        #     f.write(f"Min: {tmp_results_transposed.min():.12e}\n")
        #     f.write(f"Max: {tmp_results_transposed.max():.12e}\n")
        #     f.write(f"Mean: {tmp_results_transposed.mean():.12e}\n")
        #     f.write(f"Std: {tmp_results_transposed.std():.12e}\n\n")
        #     f.write(f"First ensemble, all receptors, first 5 timesteps:\n")
        #     for r in range(tmp_results_transposed.shape[1]):
        #         f.write(f"\nReceptor {r}:\n")
        #         for t in range(min(5, tmp_results_transposed.shape[2])):
        #             f.write(f"  T{t:3d}: {tmp_results_transposed[0, r, t]:.12e}\n")
        # print(f"[Model] Saved iteration {self._iteration_counter} ensemble observations to logs2/dev/")

        # Reshape for EKI: Need transpose to match reference final order
        # tmp_results is [ensemble, timestep, receptor] from C++
        # EKI expects [R0_T0...T23, R1_T0...T23, R2_T0...T23]
        for ens in range(tmp_states.shape[1]):
            # tmp_results[ens] is shape (num_timesteps, num_receptors) = (24, 3)
            # Transpose to (num_receptors, num_timesteps) = (3, 24) then flatten
            ensemble_obs = tmp_results[ens].T  # Now (num_receptors, num_timesteps)
            model_obs_list.append(np.asarray(ensemble_obs).reshape(-1))

        del tmp_results
        model_obs = np.asarray(model_obs_list).T
        return model_obs

    def read_initial_observations(self):
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

        # # # Save initial observation to log
        # np.save('/home/jrpark/ekitest2/logs/development/initial_observation.npy', self.obs)
        # print(f"[Model] Saved initial_observation to logs/development/initial_observation.npy")

        # Recalculate error matrices with actual observation values
        self.obs_err = np.diag((np.floor(self.obs * 0) + np.ones([len(self.obs)])*self.nreceptor_err))
        self.obs_MDA = np.diag((np.floor(self.obs * 0) + np.ones([len(self.obs)])*self.nreceptor_MDA))

    def get_ob(self, time):
        self.obs_err = (self.obs * self.obs_err + self.obs_MDA)**2   # [obs_rel_std(rate) * true_obs + obs_abs_std]**2
        return self.obs, self.obs_err

    def predict(self, state, time):
        state = np.zeros([self.nstate, self.sample])
        return state

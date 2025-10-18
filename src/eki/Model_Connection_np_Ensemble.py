import numpy as np
from copy import deepcopy

from eki_shm_config import (
    load_config_from_shared_memory,
    eki_config,
    receive_gamma_dose_matrix_shm_wrapper,
    send_tmp_states_shm
)
from eki_debug_logger import (
    save_prior_state,
    save_initial_observation,
    save_prior_ensemble,
    save_ensemble_states_sent,
    save_ensemble_observations_received
)

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

desired_gpu_index_cupy = 0

# Forward model interface
class Model(object):
    """
    Forward model interface for LDM-EKI ensemble simulations.

    Manages IPC communication with C++ LDM code and ensemble generation.

    Attributes:
        obs (ndarray): True observations from initial simulation
        obs_err (ndarray): Observation error standard deviations
        num_ensemble (int): Number of ensemble members
    """
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

        # Save prior state to debug logs (if enabled)
        save_prior_state(self.state_init)

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

            # Save initial observation to debug logs (if enabled)
            save_initial_observation(self.obs)

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
        """
        Generate prior ensemble using Gaussian sampling.

        Samples emission rates from Gaussian distribution around prior mean.
        Applies absolute value to ensure non-negative emissions.

        Args:
            prior_emission (float): Prior mean emission rate
            prior_std (float): Prior standard deviation
            num_states (int): Number of emission timesteps

        Returns:
            ndarray: Ensemble states (num_states × num_ensemble) with all positive values
        """
        state = np.empty([self.nstate, self.sample])
        for i in range(self.nstate):
            # Use abs() to prevent negative initial values (same as reference)
            state[i, :] = np.abs(np.random.normal(self.state_init[i], self.state_std[i], self.sample))

        # Save prior ensemble to debug logs (if enabled)
        save_prior_ensemble(state)

        return state
    

    def state_to_ob(self, state):
        """
        Convert ensemble states to observations via forward model (C++ LDM).

        Sends ensemble states to C++ via shared memory, waits for simulation,
        reads back ensemble observations.

        Args:
            state (ndarray): Ensemble emission states (num_states × num_ensemble)

        Returns:
            ndarray: Ensemble observations (num_receptors × num_timesteps × num_ensemble)

        Note:
            Uses IPC polling with 1-second intervals. Deletes stale observation files
            before simulation to prevent reading outdated data.
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

        if not hasattr(self, '_iteration_counter'):
            self._iteration_counter = 0
        self._iteration_counter += 1

        # Save ensemble states being sent to debug logs (if enabled)
        save_ensemble_states_sent(self._iteration_counter, tmp_states)

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

        # Save ensemble observations received to debug logs (if enabled)
        save_ensemble_observations_received(self._iteration_counter, tmp_results)

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

        # Save initial observation to debug logs (if enabled)
        save_initial_observation(self.obs)

        # Recalculate error matrices with actual observation values
        self.obs_err = np.diag((np.floor(self.obs * 0) + np.ones([len(self.obs)])*self.nreceptor_err))
        self.obs_MDA = np.diag((np.floor(self.obs * 0) + np.ones([len(self.obs)])*self.nreceptor_MDA))

    def get_ob(self, time):
        self.obs_err = (self.obs * self.obs_err + self.obs_MDA)**2   # [obs_rel_std(rate) * true_obs + obs_abs_std]**2
        return self.obs, self.obs_err

    def predict(self, state, time):
        state = np.zeros([self.nstate, self.sample])
        return state

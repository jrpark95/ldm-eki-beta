/******************************************************************************
 * @file ldm_func_output.cu
 * @brief Observation collection and output management for EKI data assimilation
 *
 * This module implements the receptor-based observation system that is central
 * to the EKI (Ensemble Kalman Inversion) framework. It manages:
 *
 * Observation Collection:
 * - Receptor locations: Virtual monitoring stations at specific lat/lon coordinates
 * - Capture radius: Spatial tolerance for particle-receptor matching (degrees)
 * - Time accumulation: Dose integrated over specified observation intervals
 * - Multi-nuclide support: Gamma dose from all nuclides weighted by DCF
 *
 * GPU Kernels:
 * - compute_eki_receptor_dose: Single-mode observation kernel
 * - compute_eki_receptor_dose_ensemble: Ensemble-mode parallel observation kernel
 * - Kernels accumulate dose contributions from particles within capture radius
 * - Results stored in 3D arrays: [ensemble][timestep][receptor] (ensemble mode)
 *                                 or 2D: [timestep][receptor] (single mode)
 *
 * Data Structures:
 * - eki_observations: Host-side observation matrix for single mode
 * - eki_ensemble_observations: Host-side observation tensor for ensemble mode
 * - d_ensemble_dose: GPU-side accumulation buffer (persistent across timesteps)
 * - d_receptor_lats/lons: GPU-side receptor locations
 *
 * Accumulation Strategy:
 * - Kernels called EVERY timestep to accumulate dose into time_idx slots
 * - Timesteps 1-9 → time_idx 0, timesteps 10-18 → time_idx 1, etc.
 * - Results copied to host only at observation boundaries
 * - GPU memory reset between EKI iterations but NOT between timesteps
 *
 * Grid Receptor Mode (Optional):
 * - Regular grid of receptors for spatial field reconstruction
 * - Used for debugging observation system and validation
 * - Generates CSV time series for each grid point
 * - Separate code path from EKI observation system
 *
 * Logging Format:
 * - [EKI_OBS] tags for single-mode observations (parsed by Python visualization)
 * - [EKI_ENSEMBLE_OBS] tags for ensemble-mode observations
 * - Scientific notation for dose values, particle counts for verification
 *
 * @note Observation accumulation happens in GPU memory, minimizing host-device transfers
 * @note Memory layout MUST match Python expectations (row-major flattening for shared memory)
 *
 * @see ldm_kernels_eki.cu for GPU kernel implementations
 * @see ldm_eki_writer.cu for shared memory IPC to Python
 * @see src/eki/eki_ipc_reader.py for Python observation reader
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/

#include "../core/ldm.cuh"
#include "ldm_func_output.cuh"
#include "../colors.h"
#include "../debug/kernel_error_collector.cuh"
#include "../core/params.hpp"

// Test function to verify g_log_file works across compilation units
void test_g_logonly_from_output_module() {
    extern std::ofstream* g_log_file;
    if (g_log_file && g_log_file->is_open()) {
        *g_log_file << "[TEST_FROM_OUTPUT_MODULE] g_log_file accessible from ldm_func_output.cu\n";
        g_log_file->flush();
    }
}

/******************************************************************************
 * @brief Start high-resolution performance timer
 *
 * Records the current time point using std::chrono for later performance
 * measurement. Used for profiling simulation sections.
 *
 * @post timerStart member variable set to current time
 *
 * @see stopTimer() to compute and report elapsed time
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::startTimer(){

        timerStart = std::chrono::high_resolution_clock::now();
    }

/******************************************************************************
 * @brief Stop performance timer and report elapsed time
 *
 * Computes time difference from last startTimer() call and outputs
 * elapsed time in seconds to stdout.
 *
 * @pre startTimer() must have been called previously
 * @post Elapsed time printed to stdout
 *
 * @note Time resolution: microseconds (1e-6 seconds)
 *
 * @see startTimer() to begin timing interval
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::stopTimer(){

        timerEnd = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(timerEnd - timerStart);
        std::cout << "Elapsed time: " << duration.count()/1.0e6 << " seconds" << std::endl;
    }

/******************************************************************************
 * @brief Initialize EKI observation system and allocate GPU memory
 *
 * Sets up the receptor-based observation collection system for EKI data
 * assimilation. This function must be called once before running EKI
 * simulations.
 *
 * Initialization Steps:
 * 1. Validate receptor locations loaded from eki_settings.txt
 * 2. Allocate GPU memory for receptor coordinates (d_receptor_lats/lons)
 * 3. Allocate GPU memory for dose accumulation (d_receptor_dose)
 * 4. Allocate GPU memory for particle count tracking
 * 5. Copy receptor locations from host to GPU
 * 6. Initialize dose/count arrays to zero
 * 7. Clear host-side observation storage vectors
 * 8. For ensemble mode: Allocate 3D ensemble dose arrays
 *
 * Memory Layout (Ensemble Mode):
 * - d_ensemble_dose: [ensemble_size × num_receptors × num_timesteps] floats
 * - d_ensemble_particle_count: Same dimensions, int type
 * - Host storage: eki_ensemble_observations[ens][time][receptor]
 *
 * Memory Layout (Single Mode):
 * - d_receptor_dose: [num_receptors] floats (accumulated per timestep)
 * - Host storage: eki_observations[time][receptor] (2D)
 *
 * @pre g_eki.receptor_locations must be loaded from config file
 * @pre g_eki.num_receptors > 0
 * @pre GPU device available and selected
 *
 * @post GPU memory allocated for observation system
 * @post Receptor locations transferred to GPU
 * @post Host observation storage initialized
 * @post System ready for computeReceptorObservations() calls
 *
 * @note Call once per simulation, NOT per EKI iteration
 * @note For EKI iterations: Use resetEKIObservationSystemForNewIteration() instead
 * @note Memory persists until cleanupEKIObservationSystem() called
 *
 * @warning Exits if receptor locations not loaded
 * @warning Requires sufficient GPU memory (proportional to ensemble_size × num_receptors × num_timesteps)
 *
 * @see cleanupEKIObservationSystem() for memory deallocation
 * @see resetEKIObservationSystemForNewIteration() for clearing between EKI iterations
 * @see computeReceptorObservations() for observation collection
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::initializeEKIObservationSystem() {
    std::cout << "Initializing observation system..." << std::endl;
    
    if (g_eki.receptor_locations.empty()) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET
                  << "No receptor locations loaded for EKI observation system" << std::endl;
        return;
    }
    
    int num_receptors = g_eki.num_receptors;
    
    // Allocate GPU memory for receptor data
    cudaMalloc(&d_receptor_lats, num_receptors * sizeof(float));
    cudaMalloc(&d_receptor_lons, num_receptors * sizeof(float));
    cudaMalloc(&d_receptor_dose, num_receptors * sizeof(float));
    cudaMalloc(&d_receptor_particle_count, num_receptors * sizeof(int));
    
    // Prepare host arrays
    std::vector<float> host_lats(num_receptors);
    std::vector<float> host_lons(num_receptors);
    
    for (int i = 0; i < num_receptors; i++) {
        host_lats[i] = g_eki.receptor_locations[i].first;   // latitude
        host_lons[i] = g_eki.receptor_locations[i].second;  // longitude
    }

#ifdef DEBUG
    // Debug: Print receptor locations being sent to GPU
    std::cout << Color::YELLOW << "[DEBUG] " << Color::RESET << "Receptor locations sent to GPU:" << std::endl;
    for (int i = 0; i < num_receptors; i++) {
        std::cout << "  Receptor " << (i+1) << ": (" << host_lats[i] << ", " << host_lons[i] << ")" << std::endl;
    }
#endif

    // Copy receptor locations to GPU
    cudaMemcpy(d_receptor_lats, host_lats.data(), num_receptors * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_receptor_lons, host_lons.data(), num_receptors * sizeof(float), cudaMemcpyHostToDevice);
    
    // Initialize dose accumulation
    cudaMemset(d_receptor_dose, 0, num_receptors * sizeof(float));
    cudaMemset(d_receptor_particle_count, 0, num_receptors * sizeof(int));

    // Initialize observation storage
    eki_observations.clear();
    eki_particle_counts.clear();
    eki_observation_count = 0;

    // Ensemble mode: Allocate or reset ensemble dose memory
    if (is_ensemble_mode) {
        int ensemble_dose_size = ensemble_size * g_eki.num_receptors * ensemble_num_states;

        if (d_ensemble_dose == nullptr) {
            // First time allocation
            cudaMalloc(&d_ensemble_dose, ensemble_dose_size * sizeof(float));
            cudaMalloc(&d_ensemble_particle_count, ensemble_dose_size * sizeof(int));

            std::cout << "Ensemble mode: Allocating GPU memory for ensemble dose" << std::endl;
            std::cout << "Ensemble dose size: " << ensemble_size << " × "
                      << g_eki.num_receptors << " × " << ensemble_num_states
                      << " = " << ensemble_dose_size << " floats" << std::endl;
        } else {
            std::cout << "Resetting ensemble GPU memory for new iteration" << std::endl;
        }

        // ALWAYS reset to zero (both first allocation and subsequent iterations)
        cudaMemset(d_ensemble_dose, 0, ensemble_dose_size * sizeof(float));
        cudaMemset(d_ensemble_particle_count, 0, ensemble_dose_size * sizeof(int));

        // Initialize host storage for ensemble observations
        eki_ensemble_observations.clear();
        eki_ensemble_particle_counts.clear();
        eki_ensemble_observations.resize(ensemble_size);
        eki_ensemble_particle_counts.resize(ensemble_size);
        for (int ens = 0; ens < ensemble_size; ens++) {
            eki_ensemble_observations[ens].resize(ensemble_num_states);
            eki_ensemble_particle_counts[ens].resize(ensemble_num_states);
            for (int t = 0; t < ensemble_num_states; t++) {
                eki_ensemble_observations[ens][t].resize(g_eki.num_receptors, 0.0f);
                eki_ensemble_particle_counts[ens][t].resize(g_eki.num_receptors, 0);
            }
        }
        std::cout << "Host storage initialized for " << ensemble_size << " ensembles" << std::endl;
    }

    std::cout << Color::GREEN << "✓ " << Color::RESET << "Initialized for "
              << Color::BOLD << num_receptors << Color::RESET << " receptors" << std::endl;
    std::cout << "  Capture radius: " << g_eki.receptor_capture_radius << " degrees" << std::endl;
}

/******************************************************************************
 * @brief Compute receptor observations for single-mode simulation
 *
 * Collects gamma dose observations at receptor locations by launching GPU
 * kernel to accumulate contributions from nearby particles. This function
 * is called EVERY timestep to maintain cumulative dose over observation
 * periods.
 *
 * Accumulation Strategy:
 * - Timesteps 1-9 accumulate into time_idx=0 (first observation period)
 * - Timesteps 10-18 accumulate into time_idx=1 (second observation period)
 * - Timesteps 19-27 accumulate into time_idx=2 (third observation period)
 * - Formula: time_idx = (timestep - 1) / timesteps_per_observation
 *
 * Kernel Execution:
 * - compute_eki_receptor_dose() kernel called each timestep
 * - Kernel searches all particles for proximity to receptors
 * - Particles within capture_radius contribute dose (distance-weighted)
 * - Results accumulated in GPU memory (d_receptor_dose_2d)
 *
 * Host Copy Strategy:
 * - GPU results copied to host only at observation boundaries
 * - Boundaries: timestep % timesteps_per_observation == 0
 * - Example: timesteps 9, 18, 27, 36, ... trigger host copy
 * - Reduces expensive GPU-to-host transfers
 *
 * Data Structures:
 * - d_receptor_dose_2d: 2D GPU array [num_timesteps × num_receptors]
 * - eki_observations: Host vector of observation vectors
 * - eki_particle_counts: Parallel vector tracking particle counts
 * - Static variables: Persist across function calls for accumulation
 *
 * Logging:
 * - Debug logs to g_log_file (function traces, data ranges)
 * - [EKI_OBS] tags for Python visualization parser
 * - Scientific notation for dose values, particle counts
 *
 * @param[in] timestep Current simulation timestep (1-indexed, 0 skipped)
 * @param[in] currentTime Current simulation time in seconds
 *
 * @pre initializeEKIObservationSystem() must be called first
 * @pre d_part contains valid particle data on GPU
 * @pre timestep > 0 (timestep 0 is skipped)
 *
 * @post Dose accumulated in GPU memory for current time_idx
 * @post At observation boundaries: Results copied to eki_observations vector
 * @post [EKI_OBS] log entry written at observation boundaries
 *
 * @note Called every timestep but only copies to host at boundaries
 * @note Static GPU memory allocated on first call, reused thereafter
 * @note Particle count logged for verification (should increase over time)
 *
 * @warning time_idx must be < max_observations to avoid buffer overflow
 * @warning Static GPU pointers: Not thread-safe (single simulation instance only)
 *
 * @see compute_eki_receptor_dose() GPU kernel for dose calculation
 * @see computeReceptorObservations_AllEnsembles() for ensemble mode
 * @see saveEKIObservationResults() to export observations to file
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::computeReceptorObservations(int timestep, float currentTime) {
    // FIXED: Match reference code ACCUMULATION mode
    // Reference calls kernel every timestep, accumulating into different time_idx slots
    // Timesteps 1-9 accumulate into time_idx=0, timesteps 10-18 into time_idx=1, etc.

    if (timestep == 0) {
        return;  // Skip timestep 0
    }

    float eki_interval_seconds = g_eki.time_interval * 60.0f;
    int timesteps_per_observation = (int)(eki_interval_seconds / dt);

    // Calculate time_idx: CORRECT FORMULA FOR ACCUMULATION
    // timestep 1-9   → time_idx = 0 (ALL accumulate to first observation)
    // timestep 10-18 → time_idx = 1 (ALL accumulate to second observation)
    // timestep 19-27 → time_idx = 2 (ALL accumulate to third observation)
    int time_idx = (timestep - 1) / timesteps_per_observation;

    // Log-only debug: function called trace
    extern std::ofstream* g_log_file;
    if (g_log_file && g_log_file->is_open() && timestep <= 10) {
        *g_log_file << "[DEBUG] computeReceptorObservations: timestep=" << timestep
                   << ", time=" << currentTime << "s, time_idx=" << time_idx << "\n";
    }

    int max_observations = static_cast<int>(time_end / eki_interval_seconds);
    if (time_idx >= max_observations) {
        return;
    }

    int num_receptors = g_eki.num_receptors;
    int num_timesteps = max_observations;

    // First time initialization: allocate 2D GPU memory if needed
    static bool gpu_memory_allocated = false;
    static float* d_receptor_dose_2d = nullptr;
    static int* d_receptor_particle_count_2d = nullptr;

    if (!gpu_memory_allocated) {
        // Allocate 2D arrays: [num_timesteps × num_receptors] - MATCH REFERENCE CODE
        int total_size = num_timesteps * num_receptors;
        cudaMalloc(&d_receptor_dose_2d, total_size * sizeof(float));
        cudaMalloc(&d_receptor_particle_count_2d, total_size * sizeof(int));
        cudaMemset(d_receptor_dose_2d, 0, total_size * sizeof(float));
        cudaMemset(d_receptor_particle_count_2d, 0, total_size * sizeof(int));
        gpu_memory_allocated = true;

        std::cout << "Allocated 2D GPU memory: " << num_timesteps << " timesteps × "
                  << num_receptors << " receptors = " << total_size << " floats" << std::endl;
    }

    int blockSize = 256;
    int numBlocks = (nop + blockSize - 1) / blockSize;

    // Prepare KernelScalars structure
    KernelScalars ks{};
    extern int g_turb_switch;
    GridConfig grid_config = loadGridConfig();
    ks.turb_switch = g_turb_switch;
    ks.drydep = 0;  // Not used in EKI kernels
    ks.wetdep = 0;  // Not used in EKI kernels
    ks.raddecay = 0;  // Not used in EKI kernels
    ks.num_particles = nop;
    ks.is_rural = isRural ? 1 : 0;
    ks.is_pg = isPG ? 1 : 0;
    ks.is_gfs = isGFS ? 1 : 0;
    ks.delta_time = dt;
    ks.grid_start_lat = grid_config.start_lat;
    ks.grid_start_lon = grid_config.start_lon;
    ks.grid_lat_step = grid_config.lat_step;
    ks.grid_lon_step = grid_config.lon_step;
    ks.settling_vel = vsetaver;
    ks.cunningham_fac = cunningham;

    // Call kernel EVERY timestep to accumulate into correct time_idx slot
    computeReceptorDose<<<numBlocks, blockSize>>>(
        d_part,
        d_receptor_lats, d_receptor_lons,
        g_eki.receptor_capture_radius,
        d_receptor_dose_2d,
        d_receptor_particle_count_2d,
        num_receptors,
        num_timesteps,
        time_idx,   // Pass time_idx to kernel
        nop,        // Pass number of particles instead of using d_nop
        time_end,   // Pass time_end instead of using d_time_end
        1.0f,       // DCF
        ks          // KernelScalars
    );

    cudaDeviceSynchronize();
    CHECK_KERNEL_ERROR();

    // Log-only debug: kernel execution completed (single mode)
    if (g_log_file && g_log_file->is_open() && timestep % timesteps_per_observation == 0) {
        *g_log_file << "[DEBUG] Single mode kernel executed for time_idx=" << time_idx
                   << " (observation boundary at timestep " << timestep << ")\n";
        *g_log_file << "  Particle count: " << nop << "\n";
        *g_log_file << "  Receptor count: " << num_receptors << "\n";
    }

    // Only copy results at observation boundaries (timesteps 9, 18, 27, ...)
    if (timestep % timesteps_per_observation == 0) {
        // Copy accumulated results for this time_idx
        std::vector<float> host_dose(num_receptors);
        std::vector<int> host_particle_count(num_receptors);

        for (int r = 0; r < num_receptors; r++) {
            // MATCH REFERENCE CODE: [timestep][receptor] layout
            int idx = time_idx * num_receptors + r;
            cudaMemcpy(&host_dose[r], &d_receptor_dose_2d[idx], sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&host_particle_count[r], &d_receptor_particle_count_2d[idx], sizeof(int), cudaMemcpyDeviceToHost);
        }

        // Store observations
        // Note: time_idx is already correct (0-indexed), just push directly
        eki_observations.push_back(host_dose);
        eki_particle_counts.push_back(host_particle_count);
        eki_observation_count = time_idx;
        // Store time for the PREVIOUS period (matching reference)
        eki_observation_times.push_back(currentTime - eki_interval_seconds);

        // Log-only debug: show captured data statistics
        if (g_log_file && g_log_file->is_open()) {
            *g_log_file << "[DEBUG] Captured data for time_idx=" << time_idx << ":\n";
            *g_log_file << "  Doses: ";
            for (int r = 0; r < num_receptors; r++) {
                *g_log_file << host_dose[r] << " ";
            }
            *g_log_file << "\n  Particle counts: ";
            for (int r = 0; r < num_receptors; r++) {
                *g_log_file << host_particle_count[r] << " ";
            }
            *g_log_file << "\n";
        }

        // Output observation data for Python visualization parser (log file only, not terminal)
        if (g_log_file && g_log_file->is_open()) {
            *g_log_file << "[EKI_OBS] Observation " << (time_idx + 1)
                      << " at t=" << (int)currentTime << "s:";
            for (int r = 0; r < num_receptors; r++) {
                *g_log_file << " R" << (r+1) << "="
                          << std::scientific << host_dose[r]
                          << "(" << host_particle_count[r] << "p)";
            }
            *g_log_file << std::endl;
        }
    }
}

void LDM::saveEKIObservationResults() {
    if (eki_observations.empty()) {
        std::cout << "No observations to save" << std::endl;
        return;
    }

    std::cout << "Saving " << eki_observations.size()
              << " observations for " << g_eki.num_receptors << " receptors" << std::endl;
    
    std::ofstream file("logs/eki_receptor_observations.txt");
    if (!file.is_open()) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET
                  << "Could not open logs/eki_receptor_observations.txt for writing" << std::endl;
        return;
    }
    
    // Write header
    file << "# EKI Receptor Observation Results\n";
    file << "# Time interval: " << g_eki.time_interval << " minutes\n";
    file << "# Capture radius: " << g_eki.receptor_capture_radius << " degrees\n";
    file << "# Receptor locations:\n";
    
    for (int r = 0; r < g_eki.num_receptors; r++) {
        file << "#   Receptor " << (r+1) << ": (" 
             << g_eki.receptor_locations[r].first << ", " 
             << g_eki.receptor_locations[r].second << ")\n";
    }
    
    file << "#\n";
    file << "# Format: Time(min)";
    for (int r = 0; r < g_eki.num_receptors; r++) {
        file << "\tReceptor" << (r+1) << "_Dose(Sv)";
    }
    file << "\n";
    
    // Write data
    for (size_t t = 0; t < eki_observations.size(); t++) {
        // Use actual observation time if available, otherwise estimate
        int time_minutes;
        if (t < eki_observation_times.size()) {
            time_minutes = (int)round(eki_observation_times[t] / 60.0f);  // Convert seconds to integer minutes
        } else {
            time_minutes = (int)(t * g_eki.time_interval);  // Fallback to expected time
        }
        file << time_minutes;
        
        for (int r = 0; r < g_eki.num_receptors; r++) {
            file << "\t" << std::scientific << eki_observations[t][r];
        }
        file << "\n";
    }
    
    file.close();
    std::cout << Color::GREEN << "✓ " << Color::RESET
              << "Results saved to logs/eki_receptor_observations.txt" << std::endl;
}

bool LDM::writeEKIObservationsToSharedMemory(void* writer_ptr) {
    std::cout << Color::CYAN << "[IPC] " << Color::RESET
              << "writeEKIObservationsToSharedMemory called (implementation in main_eki.cu)" << std::endl;
    // Implementation moved to main_eki.cu to avoid header dependencies
    return true;
}

/******************************************************************************
 * @brief Compute receptor observations for all ensemble members in parallel
 *
 * Ensemble-mode variant of computeReceptorObservations() that processes
 * multiple ensemble simulations simultaneously. Each ensemble member has
 * independent particles that contribute to separate observation channels.
 *
 * Ensemble Organization:
 * - Total particles: nop × ensemble_size (e.g., 10000 × 100 = 1M particles)
 * - Particle layout: First nop particles = ensemble 0, next nop = ensemble 1, etc.
 * - Each ensemble has separate emission time series (perturbed source terms)
 * - Observation kernel processes all ensembles in single GPU launch
 *
 * Accumulation Strategy:
 * - Same time_idx formula as single mode: (timestep - 1) / timesteps_per_observation
 * - Kernel called EVERY timestep to accumulate into time_idx slots
 * - GPU memory layout: [ensemble][timestep][receptor] (3D array)
 * - Results copied to host only at observation boundaries
 *
 * Kernel Execution:
 * - compute_eki_receptor_dose_ensemble() kernel called each timestep
 * - Kernel uses particle.ensemble_id to route contributions
 * - Each ensemble accumulates independently in parallel
 * - Single kernel launch processes all ensembles (efficient GPU utilization)
 *
 * Memory Management:
 * - d_ensemble_dose: [ensemble_size × num_timesteps × num_receptors] floats
 * - Persistent across timesteps within iteration
 * - Reset to zero at start of each EKI iteration
 * - Emergency allocation if not initialized (should not happen)
 *
 * Data Structures:
 * - eki_ensemble_observations[ens][time][receptor]: Host storage (3D)
 * - eki_ensemble_particle_counts[ens][time][receptor]: Parallel counts (3D)
 * - Index mapping: idx = ens*(TIME*RECEPT) + time_idx*RECEPT + receptor
 *
 * Logging:
 * - [EKI_ENSEMBLE_OBS] tags for Python parser
 * - Mean particle count across ensembles reported
 * - Statistics (first ensemble, mean) logged for verification
 *
 * @param[in] timestep Current simulation timestep (1-indexed, 0 skipped)
 * @param[in] currentTime Current simulation time in seconds
 * @param[in] num_ensembles Number of ensemble members (e.g., 100)
 * @param[in] num_timesteps Number of observation time periods (e.g., 12)
 *
 * @pre initializeEKIObservationSystem() called with is_ensemble_mode=true
 * @pre d_ensemble_dose allocated and zeroed
 * @pre Particles initialized with ensemble_id field set
 * @pre h_part.size() == nop * num_ensembles
 *
 * @post Dose accumulated for all ensembles in GPU memory
 * @post At boundaries: Results copied to eki_ensemble_observations
 * @post [EKI_ENSEMBLE_OBS] log entry with mean particle counts
 *
 * @note Parallel processing: All ensembles computed in single kernel launch
 * @note Memory efficiency: 3D GPU array reused across timesteps
 * @note Host copy only at boundaries to minimize transfer overhead
 *
 * @warning Requires d_ensemble_dose pre-allocated (emergency allocation as fallback)
 * @warning Particle ensemble_id must be correctly set during initialization
 * @warning GPU memory proportional to ensemble_size (can be very large)
 *
 * @see compute_eki_receptor_dose_ensemble() GPU kernel
 * @see initializeParticlesEKI_AllEnsembles() for particle setup
 * @see computeReceptorObservations() for single-mode variant
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::computeReceptorObservations_AllEnsembles(int timestep, float currentTime, int num_ensembles, int num_timesteps) {
    // FIXED: Match reference code ACCUMULATION mode for ensemble
    // Reference calls kernel every timestep, accumulating into different time_idx slots

    if (timestep == 0) {
        return;  // Skip timestep 0
    }

    float eki_interval_seconds = g_eki.time_interval * 60.0f;
    int timesteps_per_observation = (int)(eki_interval_seconds / dt);

    // Calculate time_idx: CORRECT FORMULA FOR ACCUMULATION
    // timestep 1-9   → time_idx = 0 (ALL accumulate to first observation)
    // timestep 10-18 → time_idx = 1 (ALL accumulate to second observation)
    // timestep 19-27 → time_idx = 2 (ALL accumulate to third observation)
    int time_idx = (timestep - 1) / timesteps_per_observation;

    // Log-only debug: function called trace (ensemble mode)
    extern std::ofstream* g_log_file;
    if (g_log_file && g_log_file->is_open() && timestep <= 10) {
        *g_log_file << "[DEBUG] computeReceptorObservations_AllEnsembles: timestep=" << timestep
                   << ", time=" << currentTime << "s, time_idx=" << time_idx << "\n";
    }

    if (time_idx >= num_timesteps) {
        return;
    }

    int num_receptors = g_eki.num_receptors;
    int total_particles = h_part.size();

    // Verify ensemble dose memory is initialized
    if (d_ensemble_dose == nullptr) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET
                  << "Ensemble dose memory not initialized!" << std::endl;
        std::cerr << "         This should have been allocated in initializeEKIObservationSystem()" << std::endl;
        std::cerr << "         Attempting emergency allocation..." << std::endl;

        int ensemble_dose_size = num_ensembles * num_receptors * num_timesteps;
        cudaMalloc(&d_ensemble_dose, ensemble_dose_size * sizeof(float));
        cudaMalloc(&d_ensemble_particle_count, ensemble_dose_size * sizeof(int));
        cudaMemset(d_ensemble_dose, 0, ensemble_dose_size * sizeof(float));
        cudaMemset(d_ensemble_particle_count, 0, ensemble_dose_size * sizeof(int));

        std::cout << Color::YELLOW << "[ENSEMBLE] " << Color::RESET
                  << "Emergency GPU memory allocation: "
                  << num_ensembles << "×" << num_receptors << "×" << num_timesteps
                  << " = " << ensemble_dose_size << " floats" << std::endl;

        // Initialize host storage
        eki_ensemble_observations.resize(num_ensembles);
        eki_ensemble_particle_counts.resize(num_ensembles);
        for (int ens = 0; ens < num_ensembles; ens++) {
            eki_ensemble_observations[ens].resize(num_timesteps);
            eki_ensemble_particle_counts[ens].resize(num_timesteps);
            for (int t = 0; t < num_timesteps; t++) {
                eki_ensemble_observations[ens][t].resize(num_receptors, 0.0f);
                eki_ensemble_particle_counts[ens][t].resize(num_receptors, 0);
            }
        }
    }

    int blockSize = 256;
    int numBlocks = (total_particles + blockSize - 1) / blockSize;

    // Calculate particles per ensemble
    int particles_per_ensemble = total_particles / num_ensembles;

    // Prepare KernelScalars structure
    KernelScalars ks{};
    extern int g_turb_switch;
    GridConfig grid_config = loadGridConfig();
    ks.turb_switch = g_turb_switch;
    ks.drydep = 0;  // Not used in EKI kernels
    ks.wetdep = 0;  // Not used in EKI kernels
    ks.raddecay = 0;  // Not used in EKI kernels
    ks.num_particles = particles_per_ensemble;
    ks.is_rural = isRural ? 1 : 0;
    ks.is_pg = isPG ? 1 : 0;
    ks.is_gfs = isGFS ? 1 : 0;
    ks.delta_time = dt;
    ks.grid_start_lat = grid_config.start_lat;
    ks.grid_start_lon = grid_config.start_lon;
    ks.grid_lat_step = grid_config.lat_step;
    ks.grid_lon_step = grid_config.lon_step;
    ks.settling_vel = vsetaver;
    ks.cunningham_fac = cunningham;

    // Call kernel EVERY timestep to accumulate into correct time_idx slot
    computeReceptorDoseEnsemble<<<numBlocks, blockSize>>>(
        d_part,
        d_receptor_lats, d_receptor_lons,
        g_eki.receptor_capture_radius,
        d_ensemble_dose,
        d_ensemble_particle_count,
        num_ensembles,
        num_receptors,
        num_timesteps,
        time_idx,              // Pass time_idx to kernel
        total_particles,
        particles_per_ensemble, // Pass particles per ensemble instead of using d_nop
        time_end,               // Pass time_end instead of using d_time_end
        1.0f,                   // DCF
        ks                      // KernelScalars
    );

    cudaDeviceSynchronize();
    CHECK_KERNEL_ERROR();

    // Log-only debug: kernel execution completed (ensemble mode)
    if (g_log_file && g_log_file->is_open() && timestep % timesteps_per_observation == 0) {
        *g_log_file << "[DEBUG] Ensemble kernel executed for time_idx=" << time_idx
                   << " (observation boundary at timestep " << timestep << ")\n";
        *g_log_file << "  Ensemble count: " << num_ensembles << "\n";
        *g_log_file << "  Total particles: " << total_particles << "\n";
        *g_log_file << "  Receptor count: " << num_receptors << "\n";
    }

    // Only copy results at observation boundaries (timesteps 9, 18, 27, ...)
    if (timestep % timesteps_per_observation == 0) {
        // Copy results back to host
        int ensemble_dose_size = num_ensembles * num_receptors * num_timesteps;
        std::vector<float> host_ensemble_dose(ensemble_dose_size);
        std::vector<int> host_ensemble_particle_count(ensemble_dose_size);
        cudaMemcpy(host_ensemble_dose.data(), d_ensemble_dose,
                   ensemble_dose_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_ensemble_particle_count.data(), d_ensemble_particle_count,
                   ensemble_dose_size * sizeof(int), cudaMemcpyDeviceToHost);

        // Store in structured format - MATCH REFERENCE CODE
        // GPU layout: [ensemble][timestep][receptor]
        for (int ens = 0; ens < num_ensembles; ens++) {
            for (int r = 0; r < num_receptors; r++) {
                // Reference index: ens * (TIME * RECEPT) + r + time_idx * RECEPT
                int idx = ens * (num_timesteps * num_receptors) +
                         time_idx * num_receptors +
                         r;
                eki_ensemble_observations[ens][time_idx][r] = host_ensemble_dose[idx];
                eki_ensemble_particle_counts[ens][time_idx][r] = host_ensemble_particle_count[idx];
            }
        }

        // Log-only debug: show ensemble statistics
        if (g_log_file && g_log_file->is_open()) {
            *g_log_file << "[DEBUG] Ensemble captured data for time_idx=" << time_idx << ":\n";

            // Show first ensemble's values as sample
            *g_log_file << "  First ensemble doses: ";
            for (int r = 0; r < num_receptors; r++) {
                *g_log_file << eki_ensemble_observations[0][time_idx][r] << " ";
            }
            *g_log_file << "\n  First ensemble counts: ";
            for (int r = 0; r < num_receptors; r++) {
                *g_log_file << eki_ensemble_particle_counts[0][time_idx][r] << " ";
            }
            *g_log_file << "\n";

            // Calculate and show mean values across all ensembles
            *g_log_file << "  Mean doses across ensembles: ";
            for (int r = 0; r < num_receptors; r++) {
                double mean_dose = 0.0;
                for (int ens = 0; ens < num_ensembles; ens++) {
                    mean_dose += eki_ensemble_observations[ens][time_idx][r];
                }
                mean_dose /= num_ensembles;
                *g_log_file << mean_dose << " ";
            }
            *g_log_file << "\n";
        }

        // Output ensemble average particle counts for Python visualization parser (log file only, not terminal)
        if (g_log_file && g_log_file->is_open()) {
            *g_log_file << "[EKI_ENSEMBLE_OBS] obs" << (time_idx + 1)
                      << " at t=" << (int)currentTime << "s:";
            for (int r = 0; r < num_receptors; r++) {
                // Calculate mean particle count for this receptor across all ensembles
                double mean_count = 0.0;
                for (int ens = 0; ens < num_ensembles; ens++) {
                    mean_count += eki_ensemble_particle_counts[ens][time_idx][r];
                }
                mean_count /= num_ensembles;
                *g_log_file << " R" << (r+1) << "=" << (int)mean_count << "p";
            }
            *g_log_file << std::endl;
        }
    }
}

void LDM::cleanupEKIObservationSystem() {
    if (d_receptor_lats) {
        cudaFree(d_receptor_lats);
        d_receptor_lats = nullptr;
    }
    if (d_receptor_lons) {
        cudaFree(d_receptor_lons);
        d_receptor_lons = nullptr;
    }
    if (d_receptor_dose) {
        cudaFree(d_receptor_dose);
        d_receptor_dose = nullptr;
    }
    if (d_receptor_particle_count) {
        cudaFree(d_receptor_particle_count);
        d_receptor_particle_count = nullptr;
    }
    if (d_ensemble_dose) {
        cudaFree(d_ensemble_dose);
        d_ensemble_dose = nullptr;
    }
    if (d_ensemble_particle_count) {
        cudaFree(d_ensemble_particle_count);
        d_ensemble_particle_count = nullptr;
    }

    eki_observations.clear();
    eki_particle_counts.clear();
    eki_ensemble_observations.clear();
    eki_ensemble_particle_counts.clear();
    eki_observation_count = 0;

    std::cout << Color::GREEN << "✓ " << Color::RESET << "Cleanup completed" << std::endl;
}

void LDM::resetEKIObservationSystemForNewIteration() {
    std::cout << "Resetting observation system for new iteration..." << std::endl;

    // Reset GPU memory without deallocating
    if (is_ensemble_mode && d_ensemble_dose != nullptr) {
        int ensemble_dose_size = ensemble_size * g_eki.num_receptors * ensemble_num_states;
        cudaMemset(d_ensemble_dose, 0, ensemble_dose_size * sizeof(float));
        cudaMemset(d_ensemble_particle_count, 0, ensemble_dose_size * sizeof(int));

        std::cout << "Reset GPU memory: " << ensemble_dose_size << " floats" << std::endl;
    }

    // Clear and reinitialize host storage
    eki_ensemble_observations.clear();
    eki_ensemble_particle_counts.clear();
    eki_ensemble_observations.resize(ensemble_size);
    eki_ensemble_particle_counts.resize(ensemble_size);

    for (int ens = 0; ens < ensemble_size; ens++) {
        eki_ensemble_observations[ens].resize(ensemble_num_states);
        eki_ensemble_particle_counts[ens].resize(ensemble_num_states);
        for (int t = 0; t < ensemble_num_states; t++) {
            eki_ensemble_observations[ens][t].resize(g_eki.num_receptors, 0.0f);
            eki_ensemble_particle_counts[ens][t].resize(g_eki.num_receptors, 0);
        }
    }

    std::cout << Color::GREEN << "✓ " << Color::RESET
              << "Host storage reinitialized for " << ensemble_size
              << " ensembles, " << ensemble_num_states << " timesteps" << std::endl;
}

void LDM::computeGridReceptorObservations(int timestep, float currentTime) {
    if (!is_grid_receptor_mode) {
        return;  // Only run in grid receptor mode
    }

    // Record observations every 60 seconds (600 timesteps with dt=100s -> every 60s = 6 timesteps)
    const int observation_interval = 6;  // Every 6 timesteps = 600s

    if (timestep == 0 || timestep % observation_interval != 0) {
        return;
    }

#ifdef DEBUG
    // Debug output
    static int obs_count = 0;
    obs_count++;
    if (obs_count <= 5 || obs_count % 10 == 0) {
        std::cout << Color::YELLOW << "[DEBUG] " << Color::RESET
                  << "Recording observation #" << obs_count
                  << " at timestep " << timestep << " (t=" << currentTime << "s)" << std::endl;
    }
#endif

    // Reset dose and particle count for this timestep
    cudaMemset(d_grid_receptor_dose, 0, grid_receptor_total * sizeof(float));
    cudaMemset(d_grid_receptor_particle_count, 0, grid_receptor_total * sizeof(int));

    // Launch GPU kernel to compute receptor doses
    // Reuse the same kernel as EKI observation system
    int blockSize = 256;
    int numBlocks = (nop + blockSize - 1) / blockSize;

    // Use receptor capture radius of 0.1 degrees (same as EKI default)
    float capture_radius = 0.1f;

    // Prepare KernelScalars structure
    KernelScalars ks{};
    extern int g_turb_switch;
    GridConfig grid_config = loadGridConfig();
    ks.turb_switch = g_turb_switch;
    ks.drydep = 0;  // Not used in EKI kernels
    ks.wetdep = 0;  // Not used in EKI kernels
    ks.raddecay = 0;  // Not used in EKI kernels
    ks.num_particles = nop;
    ks.is_rural = isRural ? 1 : 0;
    ks.is_pg = isPG ? 1 : 0;
    ks.is_gfs = isGFS ? 1 : 0;
    ks.delta_time = dt;
    ks.grid_start_lat = grid_config.start_lat;
    ks.grid_start_lon = grid_config.start_lon;
    ks.grid_lat_step = grid_config.lat_step;
    ks.grid_lon_step = grid_config.lon_step;
    ks.settling_vel = vsetaver;
    ks.cunningham_fac = cunningham;

    // Grid receptors don't need time_idx - call with num_timesteps=1, time_idx=0
    computeReceptorDose<<<numBlocks, blockSize>>>(
        d_part,
        d_grid_receptor_lats,
        d_grid_receptor_lons,
        capture_radius,
        d_grid_receptor_dose,
        d_grid_receptor_particle_count,
        grid_receptor_total,
        1,       // num_timesteps = 1 (no time dimension)
        0,       // time_idx = 0 (always use slot 0)
        nop,     // Pass number of particles instead of using d_nop
        time_end, // Pass time_end instead of using d_time_end
        1.0f,    // DCF
        ks       // KernelScalars
    );

    cudaDeviceSynchronize();
    CHECK_KERNEL_ERROR();

    // Copy results back to host
    std::vector<float> host_dose(grid_receptor_total);
    std::vector<int> host_particle_count(grid_receptor_total);

    cudaMemcpy(host_dose.data(), d_grid_receptor_dose, grid_receptor_total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_particle_count.data(), d_grid_receptor_particle_count, grid_receptor_total * sizeof(int), cudaMemcpyDeviceToHost);

    // Store observations for each receptor
    for (int r = 0; r < grid_receptor_total; r++) {
        grid_receptor_observations[r].push_back(host_dose[r]);
        grid_receptor_particle_counts[r].push_back(host_particle_count[r]);
    }

    // Store observation time
    grid_observation_times.push_back(currentTime);

#ifdef DEBUG
    // Occasional debug output showing max values
    if (obs_count <= 5 || obs_count % 10 == 0) {
        float max_dose = *std::max_element(host_dose.begin(), host_dose.end());
        int max_particles = *std::max_element(host_particle_count.begin(), host_particle_count.end());
        std::cout << Color::YELLOW << "[DEBUG] " << Color::RESET
                  << "Max dose: " << max_dose << " Sv, Max particles: " << max_particles << std::endl;
    }
#endif
}

void LDM::saveGridReceptorData() {
    if (!is_grid_receptor_mode || grid_observation_times.empty()) {
        std::cout << "No grid receptor data to save" << std::endl;
        return;
    }

    std::cout << "Saving data for " << grid_receptor_total << " receptors" << std::endl;
    std::cout << "Number of time steps: " << grid_observation_times.size() << std::endl;

    // Save each receptor's data to a separate CSV file
    for (int r = 0; r < grid_receptor_total; r++) {
        // Create filename with zero-padded receptor number
        std::ostringstream filename;
        filename << "grid_receptors/receptor_" << std::setfill('0') << std::setw(4) << r << ".csv";

        std::ofstream outfile(filename.str());
        if (!outfile.is_open()) {
            std::cerr << Color::RED << "[ERROR] " << Color::RESET
                      << "Could not open file: " << filename.str() << std::endl;
            continue;
        }

        // Write CSV header
        outfile << "Time(s),Dose(Sv),ParticleCount\n";

        // Write data for each timestep
        for (size_t t = 0; t < grid_observation_times.size(); t++) {
            outfile << grid_observation_times[t] << ","
                    << grid_receptor_observations[r][t] << ","
                    << grid_receptor_particle_counts[r][t] << "\n";
        }

        outfile.close();

        // Progress indicator
        if ((r+1) % 20 == 0 || r == grid_receptor_total - 1) {
            std::cout << "Saved " << (r+1) << "/" << grid_receptor_total << " receptor files" << std::endl;
        }
    }

    // Save grid metadata
    std::ofstream metafile("grid_receptors/grid_metadata.txt");
    if (metafile.is_open()) {
        metafile << "Grid Configuration\n";
        metafile << "==================\n";
        metafile << "Grid count: " << grid_count << " (in each direction)\n";
        metafile << "Grid spacing: " << grid_spacing << " degrees\n";
        metafile << "Total receptors: " << grid_receptor_total << "\n";
        metafile << "Grid dimensions: " << (2*grid_count+1) << "×" << (2*grid_count+1) << "\n";
        metafile << "Number of time steps: " << grid_observation_times.size() << "\n";
        metafile << "Observation interval: 600 seconds (10 minutes)\n";
        metafile << "Receptor capture radius: 0.1 degrees\n";
        metafile.close();
    }

    std::cout << Color::GREEN << "✓ " << Color::RESET
              << "All receptor data saved to grid_receptors/" << std::endl;
}

void LDM::cleanupGridReceptorSystem() {
    if (!is_grid_receptor_mode) {
        return;
    }

    std::cout << "Cleaning up grid receptor system..." << std::endl;

    // Free GPU memory
    if (d_grid_receptor_lats) {
        cudaFree(d_grid_receptor_lats);
        d_grid_receptor_lats = nullptr;
    }
    if (d_grid_receptor_lons) {
        cudaFree(d_grid_receptor_lons);
        d_grid_receptor_lons = nullptr;
    }
    if (d_grid_receptor_dose) {
        cudaFree(d_grid_receptor_dose);
        d_grid_receptor_dose = nullptr;
    }
    if (d_grid_receptor_particle_count) {
        cudaFree(d_grid_receptor_particle_count);
        d_grid_receptor_particle_count = nullptr;
    }

    // Clear host storage
    grid_receptor_observations.clear();
    grid_receptor_particle_counts.clear();
    grid_observation_times.clear();

    std::cout << Color::GREEN << "✓ " << Color::RESET << "Cleanup completed" << std::endl;
}


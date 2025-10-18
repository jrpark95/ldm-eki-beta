/******************************************************************************
 * @file ldm_func_simulation.cu
 * @brief Main simulation loop and time integration for LDM particle dispersion
 *
 * This module implements the core time-stepping loop for the Lagrangian
 * Dispersion Model (LDM). It orchestrates particle advection, meteorological
 * data updates, and observation collection across three operational modes:
 * standard simulation, EKI single mode, and EKI ensemble mode.
 *
 * Key Functions:
 * - runSimulation(): Standard forward simulation with on-demand GFS data loading
 * - runSimulation_eki(): EKI-optimized simulation with preloaded meteorological data
 * - runSimulation_eki_dump(): Debug variant with detailed particle state logging
 *
 * Time Integration Logic:
 * - Fixed timestep (dt) forward Euler integration
 * - Particle activation scheduled by emission time series
 * - Meteorological data interpolated between past/future time brackets
 * - Physics kernels called each timestep: advection, turbulence, deposition, decay
 *
 * Meteorological Data Management:
 * - Standard mode: Load GFS data on-demand when simulation crosses time interval
 * - EKI mode: Use preloaded cache (g_eki_meteo) for fast ensemble iterations
 * - Automatic past/future pair selection based on current simulation time
 * - Height field synchronization with pressure-level data
 *
 * Performance Optimizations:
 * - Fixed-size progress bar with in-place terminal updates (ANSI escape codes)
 * - VTK output controlled by enable_vtk_output flag (expensive I/O)
 * - Meteorological data preloading eliminates file I/O during EKI iterations
 * - GPU synchronization only at kernel boundaries
 *
 * @note All three simulation functions share similar structure but differ in:
 *       - Meteorological data loading strategy
 *       - Particle initialization (single vs. ensemble)
 *       - Observation collection (single vs. per-ensemble)
 *       - Output verbosity and logging
 *
 * @warning Always call preloadAllEKIMeteorologicalData() before runSimulation_eki()
 * @warning VTK output disabled by default in EKI mode for performance
 *
 * @see ldm_mdata_cache.cu for meteorological data preloading
 * @see ldm_kernels_particle.cu for particle advection kernels
 * @see ldm_func_output.cu for observation collection
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/

#include "../core/ldm.cuh"
#include "ldm_func_simulation.cuh"
#include "../debug/kernel_error_collector.cuh"
#include "../colors.h"
#include <unistd.h>  // for isatty()

/******************************************************************************
 * @brief Standard simulation loop with on-demand meteorological data loading
 *
 * Executes a complete forward simulation of particle dispersion using the
 * Lagrangian approach. This is the standard operational mode used for
 * standalone simulations (not coupled with EKI).
 *
 * Simulation Workflow:
 * 1. Initialize grid mesh and deposition arrays
 * 2. Enter main time-stepping loop (until currentTime >= time_end)
 * 3. Update particle flags based on emission schedule (activation ratio)
 * 4. Call particle movement kernel (advection + turbulence + physics)
 * 5. Check if new meteorological data needs loading (GFS time interval)
 * 6. Output VTK files and diagnostics at specified intervals
 * 7. Accumulate wall-clock time and report performance
 *
 * Meteorological Data Loading:
 * - GFS data loaded on-demand when (currentTime/time_interval) > gfs_idx
 * - Asynchronous loading supported via std::future (optional)
 * - Data interpolated between past/future time brackets (flex_pres0/1, flex_unis0/1)
 *
 * Output Control:
 * - VTK particle snapshots written every freq_output timesteps
 * - Binary particle files for post-processing
 * - Concentration and nuclide ratio logging (debug builds)
 *
 * @pre Particles must be initialized (part vector populated)
 * @pre GPU memory allocated (d_part allocated via allocateGPUMemory())
 * @pre Grid configuration loaded (loadGridConfig() must succeed)
 * @pre Initial meteorological data loaded
 *
 * @post Simulation completes at time_end
 * @post Final particle positions stored in d_part (GPU) and part (host if copied)
 * @post Output files written to output/ directory
 *
 * @note Grid receptor mode: Also records observations at grid points if enabled
 * @note Progress output: Basic timestep/time reporting without fancy progress bar
 *
 * @warning High I/O cost: Frequent VTK output can dominate runtime
 * @warning Memory usage: Large particle counts require careful GPU memory management
 *
 * @see loadFlexGFSData() for on-demand meteorological data loading
 * @see move_part_by_wind_mpi() for particle advection kernel
 * @see outputParticlesBinaryMPI() for VTK output generation
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::runSimulation(){

    cudaError_t err = cudaGetLastError();

    std::future<void> gfs_future;
    bool gfs_ready = false;

    int ded;
    
    float currentTime = 0.0f;
    int threadsPerBlock = 256;
    int blocks = (h_part.size() + threadsPerBlock - 1) / threadsPerBlock;
    int timestep = 0;
    float activationRatio = 0.0;
    float t0 = 0.0;
    float totalElapsedTime = 0.0;

    GridConfig grid_config = loadGridConfig();
    float start_lat = grid_config.start_lat;
    float start_lon = grid_config.start_lon;
    float end_lat = grid_config.end_lat;
    float end_lon = grid_config.end_lon;
    float lat_step = grid_config.lat_step;
    float lon_step = grid_config.lon_step;
    
    int lat_num = static_cast<int>(round((end_lat - start_lat) / lat_step) + 1);
    int lon_num = static_cast<int>(round((end_lon - start_lon) / lon_step) + 1);

    // Grid parameters now passed via KernelScalars struct, not constant memory
    Mesh mesh(start_lat, start_lon, lat_step, lon_step, lat_num, lon_num);

    size_t meshSize = mesh.lat_count * mesh.lon_count * sizeof(float);

    float* d_dryDep = nullptr;
    float* d_wetDep = nullptr;

    cudaMalloc((void**)&d_dryDep, meshSize);
    cudaMalloc((void**)&d_wetDep, meshSize);

    cudaMemset(d_dryDep, 0, meshSize);
    cudaMemset(d_wetDep, 0, meshSize);

    FlexPres* flexpresdata = new FlexPres[(dimX_GFS + 1) * dimY_GFS * dimZ_GFS];
    FlexUnis* flexunisdata = new FlexUnis[(dimX_GFS + 1) * dimY_GFS];

    //log_first_particle_concentrations(0, 0.0f);

    while(currentTime < time_end){


        stepStart = std::chrono::high_resolution_clock::now();

        currentTime += dt;

        activationRatio = (currentTime) / time_end;
        t0 = (currentTime - static_cast<int>(currentTime/time_interval)*time_interval) / time_interval;

        updateParticleFlags<<<blocks, threadsPerBlock>>>
            (d_part, activationRatio, nop);
        cudaDeviceSynchronize();
        CHECK_KERNEL_ERROR();

        NuclideConfig* nucConfig = NuclideConfig::getInstance();

        // Populate KernelScalars for particle movement kernel
        KernelScalars ks{};
        ks.turb_switch = g_turb_switch;
        ks.drydep = g_drydep;
        ks.wetdep = g_wetdep;
        ks.raddecay = g_raddecay;
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
        ks.T_matrix = d_T_matrix;
        ks.height_levels = d_height_levels;

        advectParticles<<<blocks, threadsPerBlock>>>
        (d_part, t0, PROCESS_INDEX, d_dryDep, d_wetDep, mesh.lon_count, mesh.lat_count,
            d_surface_past,
            d_pressure_past,
            d_surface_future,
            d_pressure_future,
            ks);
        cudaDeviceSynchronize();
        CHECK_KERNEL_ERROR();

        timestep++;

        // Grid receptor debug mode: Record observations at regular intervals
        if (is_grid_receptor_mode) {
            computeGridReceptorObservations(timestep, currentTime);
        }

        // Debug: Copy and print first particle position every 5 timesteps
        if(timestep % 5 == 0) {  // Debug output every 5 timesteps
            LDMpart first_particle;
            cudaError_t err = cudaMemcpy(&first_particle, d_part, sizeof(LDMpart), cudaMemcpyDeviceToHost);
            if (err == cudaSuccess) {
                // Convert from GFS grid coordinates to geographic coordinates
                float lon = first_particle.x * 0.5f - 179.0f;
                float lat = first_particle.y * 0.5f - 90.0f;
                float z = first_particle.z;
                
                // Debug output disabled for release
                // printf("[CRAM2] Timestep %d: Particle 0: lon=%.6f, lat=%.6f, z=%.2f | Wind: u=%.6f, v=%.6f, w=%.6f m/s\n", 
                //        timestep, lon, lat, z, first_particle.u_wind, first_particle.v_wind, first_particle.w_wind);
                // fflush(stdout); // Force output
            }
        }

        if(timestep % freq_output==0){
            printf("-------------------------------------------------\n");
            printf("Time : %f\tsec\n", currentTime);
            printf("Time steps : \t%d of \t%d\n", timestep, (int)(time_end/dt));
            
            // Debug GFS loading condition at key timepoints
            if(timestep == 1080) {  // Check at 10800s (3 hours) for GFS data validation
                printf("[DEBUG] GFS condition check: currentTime=%.1f, time_interval=%d, left=%d, gfs_idx=%d, condition=%s\n", 
                       currentTime, time_interval, static_cast<int>(currentTime/time_interval), gfs_idx,
                       (static_cast<int>(currentTime/time_interval) > gfs_idx) ? "TRUE" : "FALSE");
            }

            //particle_output_ASCII(timestep);
            outputParticlesBinaryMPI(timestep);
            
            // Log concentration data for analysis
            log_first_particle_concentrations(timestep, currentTime);
            log_all_particles_nuclide_ratios(timestep, currentTime);
            log_first_particle_cram_detail(timestep, currentTime, dt);
            log_first_particle_decay_analysis(timestep, currentTime);
            
            // Export validation reference data
            exportValidationData(timestep, currentTime);
        }
        stepEnd = std::chrono::high_resolution_clock::now();
        auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(stepEnd - stepStart);
        totalElapsedTime += static_cast<double>(duration0.count()/1.0e6);

        // Check GFS loading condition every timestep 
        int left_val = static_cast<int>(currentTime/time_interval);
        if(timestep >= 1079 && timestep <= 1081) {
            printf("[DEBUG] Step %d: currentTime=%.1f, left=%d, gfs_idx=%d\n", timestep, currentTime, left_val, gfs_idx);
        }
        
        if(left_val > gfs_idx) {
            printf("[INFO] Condition met: currentTime=%.1f, time_interval=%d, left=%d, gfs_idx=%d\n", 
                   currentTime, time_interval, left_val, gfs_idx);
            loadFlexGFSData();
        }

    }

    std::cout << "Elapsed data time: " << totalElapsedTime << " seconds" << std::endl;

    
}

/******************************************************************************
 * @brief EKI-optimized simulation loop with preloaded meteorological data
 *
 * Executes particle dispersion simulation optimized for Ensemble Kalman
 * Inversion (EKI) framework. This function is called repeatedly during
 * EKI iterations, so it prioritizes speed over output verbosity.
 *
 * Key Differences from Standard runSimulation():
 * - Uses preloaded meteorological data cache (g_eki_meteo) - NO file I/O
 * - Supports both single mode (true simulation) and ensemble mode (parallel)
 * - Automatic past/future meteorological pair selection each timestep
 * - VTK output controlled by enable_vtk_output flag (default: OFF)
 * - Real-time observation collection for EKI data assimilation
 * - Fancy progress bar with in-place updates (ANSI escape codes)
 *
 * Simulation Workflow:
 * 1. Verify meteorological data preloaded (g_eki_meteo.is_initialized)
 * 2. Display configuration summary (mode, particle count, GPU config)
 * 3. Enter main time-stepping loop
 * 4. Auto-select past/future meteorological data from cache (no disk I/O)
 * 5. Copy meteorological data from cache to GPU working buffers
 * 6. Call particle movement kernel (single or ensemble variant)
 * 7. Collect receptor observations (every timestep, accumulated)
 * 8. Update progress bar (fixed-scroll in-place update)
 * 9. Optional VTK output at freq_output intervals (if enabled)
 * 10. Report total elapsed time
 *
 * Meteorological Data Management:
 * - Past index: floor(currentTime / time_interval)
 * - Future index: past_index + 1
 * - Automatic range checking and boundary handling
 * - Height field (d_height_levels) synchronized with pressure-level data
 * - Device-to-device memcpy from cache to working buffers
 *
 * Ensemble vs. Single Mode:
 * - Single mode: Use standard kernels (update_particle_flags, move_part_by_wind_mpi)
 * - Ensemble mode: Use ensemble kernels (*_ens variants)
 * - Observation collection dispatched to appropriate function variant
 *
 * Performance Optimizations:
 * - All meteorological data in GPU memory (preloaded)
 * - No file I/O during simulation loop
 * - VTK output disabled by default (expensive I/O)
 * - Fixed-size progress bar (no log spam)
 * - Debug logging only to file, not terminal
 *
 * @pre g_eki_meteo.is_initialized == true (call preloadAllEKIMeteorologicalData() first)
 * @pre Particles initialized (via initializeParticlesEKI or initializeParticlesEKI_AllEnsembles)
 * @pre GPU memory allocated (d_part, meteorological buffers)
 * @pre Observation system initialized (initializeEKIObservationSystem)
 *
 * @post Simulation completes at time_end
 * @post Observations collected in eki_observations or eki_ensemble_observations
 * @post Particle state available for inspection or copying back to host
 *
 * @note Always called after preloadAllEKIMeteorologicalData() in EKI workflow
 * @note Progress bar uses stderr for in-place updates (not logged to file)
 * @note Debug NaN checks only active in DEBUG builds (first few timesteps)
 *
 * @warning Requires preloaded meteorological data - will error if not initialized
 * @warning VTK output disabled by default - set enable_vtk_output = true to enable
 * @warning Ensemble mode requires h_part.size() = nop * ensemble_size
 *
 * @see preloadAllEKIMeteorologicalData() for meteorological cache initialization
 * @see initializeEKIObservationSystem() for observation system setup
 * @see computeReceptorObservations() for single mode observations
 * @see computeReceptorObservations_AllEnsembles() for ensemble mode observations
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::runSimulation_eki(){

    cudaError_t err = cudaGetLastError();

    int ded;

    float currentTime = 0.0f;
    int threadsPerBlock = 256;
    int blocks = (h_part.size() + threadsPerBlock - 1) / threadsPerBlock;
    int timestep = 0;
    float activationRatio = 0.0;
    float t0 = 0.0;
    float totalElapsedTime = 0.0;

    // Display simulation mode and particle configuration
    std::cout << "\n" << Color::BOLD << Color::CYAN
              << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
              << "  SIMULATION CONFIGURATION\n"
              << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
              << Color::RESET;
    if (is_ensemble_mode) {
        std::cout << "  Mode           : " << Color::MAGENTA << "ENSEMBLE" << Color::RESET << std::endl;
        std::cout << "  Particles      : " << Color::BOLD << h_part.size() << Color::RESET
                  << " (" << ensemble_size << " × " << ensemble_num_states << " states)" << std::endl;
    } else {
        std::cout << "  Mode           : SINGLE" << std::endl;
        std::cout << "  Particles      : " << Color::BOLD << h_part.size() << Color::RESET << std::endl;
    }
    std::cout << "  GPU config     : " << blocks << " blocks × " << threadsPerBlock << " threads\n" << std::endl;

    GridConfig grid_config = loadGridConfig();
    float start_lat = grid_config.start_lat;
    float start_lon = grid_config.start_lon;
    float end_lat = grid_config.end_lat;
    float end_lon = grid_config.end_lon;
    float lat_step = grid_config.lat_step;
    float lon_step = grid_config.lon_step;
    
    int lat_num = static_cast<int>(round((end_lat - start_lat) / lat_step) + 1);
    int lon_num = static_cast<int>(round((end_lon - start_lon) / lon_step) + 1);

    // Grid parameters now passed via KernelScalars struct, not constant memory
    Mesh mesh(start_lat, start_lon, lat_step, lon_step, lat_num, lon_num);

    size_t meshSize = mesh.lat_count * mesh.lon_count * sizeof(float);

    float* d_dryDep = nullptr;
    float* d_wetDep = nullptr;

    cudaMalloc((void**)&d_dryDep, meshSize);
    cudaMalloc((void**)&d_wetDep, meshSize);

    cudaMemset(d_dryDep, 0, meshSize);
    cudaMemset(d_wetDep, 0, meshSize);

    // EKI mode: Check meteorological data preloading
    if (!g_eki_meteo.is_initialized) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET
                  << "EKI meteorological data not initialized. Call preloadAllEKIMeteorologicalData() first." << std::endl;
        return;
    }
    std::cout << "\nEKI simulation starting - using preloaded meteorological data\n" << std::endl;

    std::cout << Color::BOLD << Color::CYAN
              << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
              << "  RUNNING SIMULATION\n"
              << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
              << Color::RESET << std::endl;

#ifdef DEBUG
    // === NaN check 4: Before simulation start ===
    checkParticleNaN("Before simulation start");
#endif

    //log_first_particle_concentrations(0, 0.0f);

    while(currentTime < time_end){

        stepStart = std::chrono::high_resolution_clock::now();

        currentTime += dt;

        activationRatio = (currentTime) / time_end;
        t0 = (currentTime - static_cast<int>(currentTime/time_interval)*time_interval) / time_interval;

        // EKI mode: Automatic meteorological data selection (past/future pair)
        int meteo_time_interval = Constants::time_interval;
        int past_meteo_index = static_cast<int>(currentTime / meteo_time_interval);
        int future_meteo_index = past_meteo_index + 1;
        
        // Range check and meteorological data update
        if (past_meteo_index >= 0 && past_meteo_index < g_eki_meteo.num_time_steps) {
            // Past meteorological data (d_pressure_past, unis0)
            FlexPres* past_pres_ptr;
            FlexUnis* past_unis_ptr;
            
            cudaMemcpy(&past_pres_ptr, &g_eki_meteo.d_pressure_array[past_meteo_index], 
                       sizeof(FlexPres*), cudaMemcpyDeviceToHost);
            cudaMemcpy(&past_unis_ptr, &g_eki_meteo.d_surface_array[past_meteo_index], 
                       sizeof(FlexUnis*), cudaMemcpyDeviceToHost);
            
            cudaMemcpy(d_pressure_past, past_pres_ptr,
                       g_eki_meteo.pressure_size, cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_surface_past, past_unis_ptr,
                       g_eki_meteo.surface_size, cudaMemcpyDeviceToDevice);

            // DEBUG: Verify GPU meteorological data at first timestep
            if (timestep == 0) {
                FlexPres sample_pres[3];
                // Sample at particle's expected location (source: 37.0°N, 141.0°E)
                // GFS coordinates: lon_idx = (141 + 179) / 0.5 = 640, lat_idx = (37 + 90) / 0.5 = 254
                int test_xidx = 640;  // 141°E
                int test_yidx = 254;  // 37°N
                int test_zidx = 5;    // ~1000m altitude

                int idx0 = test_xidx * dimY_GFS * dimZ_GFS + test_yidx * dimZ_GFS + test_zidx;
                int idx1 = (test_xidx+1) * dimY_GFS * dimZ_GFS + test_yidx * dimZ_GFS + test_zidx;
                int idx2 = test_xidx * dimY_GFS * dimZ_GFS + (test_yidx+1) * dimZ_GFS + test_zidx;

                cudaMemcpy(sample_pres, &d_pressure_past[idx0], sizeof(FlexPres), cudaMemcpyDeviceToHost);
                cudaMemcpy(sample_pres+1, &d_pressure_past[idx1], sizeof(FlexPres), cudaMemcpyDeviceToHost);
                cudaMemcpy(sample_pres+2, &d_pressure_past[idx2], sizeof(FlexPres), cudaMemcpyDeviceToHost);

                extern std::ofstream* g_log_file;
                if (g_log_file && g_log_file->is_open()) {
                    *g_log_file << "[GPU_METEO_VERIFY] Timestep 0 - GPU meteorological data at source location (141°E, 37°N, ~1km):\n";
                    *g_log_file << "  Grid indices: xidx=" << test_xidx << ", yidx=" << test_yidx << ", zidx=" << test_zidx << "\n";
                    *g_log_file << "  Point [" << test_xidx << "," << test_yidx << "," << test_zidx << "]: UU=" << sample_pres[0].UU << " VV=" << sample_pres[0].VV << " WW=" << sample_pres[0].WW << " m/s\n";
                    *g_log_file << "  Point [" << (test_xidx+1) << "," << test_yidx << "," << test_zidx << "]: UU=" << sample_pres[1].UU << " VV=" << sample_pres[1].VV << " WW=" << sample_pres[1].WW << " m/s\n";
                    *g_log_file << "  Point [" << test_xidx << "," << (test_yidx+1) << "," << test_zidx << "]: UU=" << sample_pres[2].UU << " VV=" << sample_pres[2].VV << " WW=" << sample_pres[2].WW << " m/s\n";
                    *g_log_file << std::flush;
                }
            }

            // Update height data as well
            g_height_levels = g_eki_meteo.h_height[past_meteo_index];

#ifdef DEBUG
            // Verify height data at first timestep only
            if (timestep == 0) {
                printf("[DEBUG] Timestep %d: Index %d height data first 5 values: ", timestep, past_meteo_index);
                for (int i = 0; i < std::min(5, (int)g_height_levels.size()); i++) {
                    printf("%.1f ", g_height_levels[i]);
                }
                printf("... %.1f\n", g_height_levels[g_height_levels.size()-1]);
            }
#endif
            
            // Copy height data to GPU memory
            cudaError_t hgt_err = cudaMemcpy(d_height_levels, g_height_levels.data(), sizeof(float) * dimZ_GFS, cudaMemcpyHostToDevice);
            if (hgt_err != cudaSuccess) {
                // Log to file only (collected by Kernel Error Collector for batch reporting)
                extern std::ofstream* g_log_file;
                if (g_log_file && g_log_file->is_open()) {
                    *g_log_file << "[ERROR] Failed to copy height data to GPU: "
                                << cudaGetErrorString(hgt_err) << "\n" << std::flush;
                }
            }
        }

        // Future meteorological data (d_pressure_future, unis1)
        if (future_meteo_index >= 0 && future_meteo_index < g_eki_meteo.num_time_steps) {
            FlexPres* future_pres_ptr;
            FlexUnis* future_unis_ptr;
            
            cudaMemcpy(&future_pres_ptr, &g_eki_meteo.d_pressure_array[future_meteo_index], 
                       sizeof(FlexPres*), cudaMemcpyDeviceToHost);
            cudaMemcpy(&future_unis_ptr, &g_eki_meteo.d_surface_array[future_meteo_index], 
                       sizeof(FlexUnis*), cudaMemcpyDeviceToHost);
            
            cudaMemcpy(d_pressure_future, future_pres_ptr, 
                       g_eki_meteo.pressure_size, cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_surface_future, future_unis_ptr, 
                       g_eki_meteo.surface_size, cudaMemcpyDeviceToDevice);
        } else {
            // Last time period: Set future data same as past data (prevent extrapolation)
            if (past_meteo_index >= 0 && past_meteo_index < g_eki_meteo.num_time_steps) {
                FlexPres* past_pres_ptr;
                FlexUnis* past_unis_ptr;
                
                cudaMemcpy(&past_pres_ptr, &g_eki_meteo.d_pressure_array[past_meteo_index], 
                           sizeof(FlexPres*), cudaMemcpyDeviceToHost);
                cudaMemcpy(&past_unis_ptr, &g_eki_meteo.d_surface_array[past_meteo_index], 
                           sizeof(FlexUnis*), cudaMemcpyDeviceToHost);
                
                cudaMemcpy(d_pressure_future, past_pres_ptr, 
                           g_eki_meteo.pressure_size, cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_surface_future, past_unis_ptr, 
                           g_eki_meteo.surface_size, cudaMemcpyDeviceToDevice);
            }
        }

#ifdef DEBUG
        // === Detailed NaN check at first timestep only ===
        if (timestep == 0) {
            printf("[DEBUG] First timestep - detailed check before kernel execution\n");
            printf("  Current time: %.1fs, t0: %.6f\n", currentTime, t0);
            printf("  Past meteo index: %d, Future meteo index: %d\n", past_meteo_index, future_meteo_index);
            checkParticleNaN("Before first kernel execution", 5);
            checkMeteoDataNaN("After meteorological data update");
        }
#endif

        // Use ensemble kernel for ensemble mode, regular kernel for single mode
        if (is_ensemble_mode) {
            // Ensemble mode: activate particles independently per ensemble
            int total_particles = h_part.size();
            int particles_per_ensemble = nop;  // Particles per ensemble (10000)
            updateParticleFlagsEnsemble<<<blocks, threadsPerBlock>>>
                (d_part, activationRatio, total_particles, particles_per_ensemble);
        } else {
            // Single mode: use standard activation
            updateParticleFlags<<<blocks, threadsPerBlock>>>
                (d_part, activationRatio, nop);
        }
        cudaDeviceSynchronize();
        CHECK_KERNEL_ERROR();

#ifdef DEBUG
        // === NaN check after first kernel ===
        if (timestep == 0) {
            checkParticleNaN("After update_particle_flags", 5);
        }
#endif

        NuclideConfig* nucConfig = NuclideConfig::getInstance();

        // Populate KernelScalars for particle movement kernel
        KernelScalars ks{};
        ks.turb_switch = g_turb_switch;
        ks.drydep = g_drydep;
        ks.wetdep = g_wetdep;
        ks.raddecay = g_raddecay;
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
        ks.T_matrix = d_T_matrix;
        ks.height_levels = d_height_levels;

        // Use ensemble kernel for ensemble mode, regular kernel for single mode
        if (is_ensemble_mode) {
            int total_particles = h_part.size();
            advectParticlesEnsemble<<<blocks, threadsPerBlock>>>
            (d_part, t0, PROCESS_INDEX, d_dryDep, d_wetDep, mesh.lon_count, mesh.lat_count,
                d_surface_past,
                d_pressure_past,
                d_surface_future,
                d_pressure_future,
                total_particles,
                ks);
        } else {
            advectParticles<<<blocks, threadsPerBlock>>>
            (d_part, t0, PROCESS_INDEX, d_dryDep, d_wetDep, mesh.lon_count, mesh.lat_count,
                d_surface_past,
                d_pressure_past,
                d_surface_future,
                d_pressure_future,
                ks);
        }
        cudaDeviceSynchronize();
        CHECK_KERNEL_ERROR();

#ifdef DEBUG
        // === Most important check: After move_part_by_wind_mpi ===
        if (timestep == 0) {
            checkParticleNaN("After move_part_by_wind_mpi (first)", 5);
        } else if (timestep <= 3) {
            checkParticleNaN("After move_part_by_wind_mpi (timestep " + std::to_string(timestep) + ")", 3);
        }
#endif

        timestep++; 

        // Debug: Copy and print first particle position every timestep (first 10 steps)
        if(timestep <= 10) {
            LDMpart first_particle;
            cudaError_t err = cudaMemcpy(&first_particle, d_part, sizeof(LDMpart), cudaMemcpyDeviceToHost);
            if (err == cudaSuccess) {
                // Convert from GFS grid coordinates to geographic coordinates
                float lon = first_particle.x * 0.5f - 179.0f;
                float lat = first_particle.y * 0.5f - 90.0f;
                float z = first_particle.z;

                extern std::ofstream* g_log_file;
                if (g_log_file && g_log_file->is_open()) {
                    *g_log_file << "[PARTICLE_POSITION] Timestep " << timestep << " (t=" << currentTime << "s): "
                                << "lon=" << lon << "°E, lat=" << lat << "°N, z=" << z << "m, "
                                << "flag=" << (first_particle.flag ? "active" : "inactive") << ", "
                                << "timeidx=" << first_particle.timeidx << ", "
                                << "activationRatio=" << activationRatio << ", "
                                << "maxActiveTimeidx=" << int(h_part.size() * activationRatio) << ", "
                                << "u_wind=" << first_particle.u_wind << ", v_wind=" << first_particle.v_wind << ", w_wind=" << first_particle.w_wind << " m/s\n"
                                << std::flush;
                }
            }
        }

        // EKI observation system - check if we should record observations (every timestep)
        if (is_ensemble_mode) {
            // Ensemble mode: compute observations for all ensemble members
            computeReceptorObservations_AllEnsembles(timestep, currentTime, ensemble_size, ensemble_num_states);
        } else {
            // Single mode: compute observations for single simulation
            computeReceptorObservations(timestep, currentTime);
        }

        if(timestep % freq_output==0){
            static bool first_time = true;

            int total_steps = (int)(time_end/dt);
            float progress = (float)timestep / total_steps * 100.0f;
            int bar_width = 40;
            int bar_filled = (int)(progress / 100.0f * bar_width);

            // Terminal: Use ANSI codes for in-place update (goes to both terminal and log via TeeStreambuf)
            if (!first_time && g_sim.fixedScrollOutput) {
                fprintf(stderr, "\033[6A");  // Move cursor up 6 lines to overwrite progress bar
            }

            fprintf(stderr, "\r-------------------------------------------------\033[K\n");
            fprintf(stderr, "\rTime: %8.1f sec │ Step: %4d/%4d [",
                   currentTime, timestep, total_steps);
            for (int i = 0; i < bar_width; i++) {
                if (i < bar_filled) fprintf(stderr, "█");
                else fprintf(stderr, "░");
            }
            fprintf(stderr, "] %.1f%%\033[K\n", progress);
            fprintf(stderr, "\rMeteo: Past=%d Future=%d │ t0=%.3f\033[K\n",
                   past_meteo_index,
                   (future_meteo_index < g_eki_meteo.num_time_steps) ? future_meteo_index : past_meteo_index,
                   t0);

            // Mode information
            if (is_ensemble_mode) {
                fprintf(stderr, "\rMode: ENSEMBLE │ Size: %d\033[K\n",
                        ensemble_size);
            } else {
                fprintf(stderr, "\rMode: SINGLE │ Particles: %d\033[K\n", (int)h_part.size());
            }

            // VTK status - show "Writing files..." if currently outputting VTK
            if (!enable_vtk_output) {
                fprintf(stderr, "\rVTK: Disabled\033[K\n");
            } else if (timestep % freq_output == 0) {
                fprintf(stderr, "\rVTK: Writing files (slow)...\033[K\n");
            } else {
                fprintf(stderr, "\rVTK: Enabled\033[K\n");
            }

            fprintf(stderr, "\r-------------------------------------------------\033[K\n");
            fflush(stderr);

            first_time = false;

            // Use ensemble output for ensemble mode, regular output for single mode
            // VTK output is DISABLED for performance - only enabled for final EKI iteration via enable_vtk_output flag
            if (enable_vtk_output) {
                if (is_ensemble_mode) {
                    outputParticlesBinaryMPI_ens(timestep);
                } else {
                    outputParticlesBinaryMPI(timestep);
                }
            }

            // // Log concentration data for analysis
            // log_first_particle_concentrations(timestep, currentTime);
            // log_all_particles_nuclide_ratios(timestep, currentTime);
            // log_first_particle_cram_detail(timestep, currentTime, dt);
            // log_first_particle_decay_analysis(timestep, currentTime);
            
            // // Export validation reference data
            // exportValidationData(timestep, currentTime);
        }
        stepEnd = std::chrono::high_resolution_clock::now();
        auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(stepEnd - stepStart);
        totalElapsedTime += static_cast<double>(duration0.count()/1.0e6);

        // EKI mode: No meteorological data loading needed (all data already in memory)
        // Previous loadFlexGFSData() call removed

    }

    // Close the progress bar with a final separator line
    fprintf(stderr, "\r-------------------------------------------------\033[K\n");
    fflush(stderr);

    std::cout << "EKI simulation completed" << std::endl;
    std::cout << "Elapsed data time: " << totalElapsedTime << " seconds" << std::endl;
}

/******************************************************************************
 * @brief EKI simulation loop with extensive debug logging and particle dumps
 *
 * Debug variant of runSimulation_eki() that calls special "dump" kernels
 * which produce detailed particle state information for troubleshooting.
 * This version is identical to runSimulation_eki() except it calls
 * move_part_by_wind_mpi_dump() and move_part_by_wind_mpi_ens_dump() kernels
 * which have additional instrumentation.
 *
 * Use Cases:
 * - Investigating numerical instabilities (NaN, Inf values)
 * - Verifying meteorological data interpolation correctness
 * - Debugging ensemble particle initialization issues
 * - Analyzing particle trajectory anomalies
 * - Validating physics model implementations
 *
 * Debug Features:
 * - Dump kernels log every particle's state at each timestep
 * - Detailed wind field interpolation diagnostics
 * - Concentration decay tracking per nuclide
 * - Position/velocity consistency checks
 *
 * Performance Impact:
 * - Significantly slower due to extensive logging (printf from GPU)
 * - Large log files generated (gigabytes for long simulations)
 * - Should only be used for short test runs or specific timestep ranges
 *
 * @pre Same preconditions as runSimulation_eki()
 * @pre Sufficient disk space for large log files
 * @pre Short simulation duration (long runs will produce excessive output)
 *
 * @post Simulation completes with extensive debug logs
 * @post Standard observations collected (same as runSimulation_eki)
 * @post Particle state available for inspection
 *
 * @note Only for debugging - not for production EKI runs
 * @note Log volume: ~1MB per timestep per 10k particles (estimate)
 * @note GPU printf buffer may overflow on very large particle counts
 *
 * @warning Very slow - only use for debugging specific issues
 * @warning Generates massive log files - monitor disk space
 * @warning May cause GPU timeouts on consumer graphics cards
 *
 * @see runSimulation_eki() for normal EKI simulation
 * @see move_part_by_wind_mpi_dump() for instrumented single-mode kernel
 * @see move_part_by_wind_mpi_ens_dump() for instrumented ensemble kernel
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::runSimulation_eki_dump(){

    cudaError_t err = cudaGetLastError();

    int ded;

    float currentTime = 0.0f;
    int threadsPerBlock = 256;
    int blocks = (h_part.size() + threadsPerBlock - 1) / threadsPerBlock;
    int timestep = 0;
    float activationRatio = 0.0;
    float t0 = 0.0;
    float totalElapsedTime = 0.0;

    // Display simulation mode and particle configuration
    std::cout << "\n" << Color::BOLD << Color::CYAN
              << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
              << "  SIMULATION CONFIGURATION\n"
              << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
              << Color::RESET;
    if (is_ensemble_mode) {
        std::cout << "  Mode           : " << Color::MAGENTA << "ENSEMBLE" << Color::RESET << std::endl;
        std::cout << "  Particles      : " << Color::BOLD << h_part.size() << Color::RESET
                  << " (" << ensemble_size << " × " << ensemble_num_states << " states)" << std::endl;
    } else {
        std::cout << "  Mode           : SINGLE" << std::endl;
        std::cout << "  Particles      : " << Color::BOLD << h_part.size() << Color::RESET << std::endl;
    }
    std::cout << "  GPU config     : " << blocks << " blocks × " << threadsPerBlock << " threads\n" << std::endl;

    GridConfig grid_config = loadGridConfig();
    float start_lat = grid_config.start_lat;
    float start_lon = grid_config.start_lon;
    float end_lat = grid_config.end_lat;
    float end_lon = grid_config.end_lon;
    float lat_step = grid_config.lat_step;
    float lon_step = grid_config.lon_step;
    
    int lat_num = static_cast<int>(round((end_lat - start_lat) / lat_step) + 1);
    int lon_num = static_cast<int>(round((end_lon - start_lon) / lon_step) + 1);

    // Grid parameters now passed via KernelScalars struct, not constant memory
    Mesh mesh(start_lat, start_lon, lat_step, lon_step, lat_num, lon_num);

    size_t meshSize = mesh.lat_count * mesh.lon_count * sizeof(float);

    float* d_dryDep = nullptr;
    float* d_wetDep = nullptr;

    cudaMalloc((void**)&d_dryDep, meshSize);
    cudaMalloc((void**)&d_wetDep, meshSize);

    cudaMemset(d_dryDep, 0, meshSize);
    cudaMemset(d_wetDep, 0, meshSize);

    // EKI mode: Check meteorological data preloading
    if (!g_eki_meteo.is_initialized) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET
                  << "EKI meteorological data not initialized. Call preloadAllEKIMeteorologicalData() first." << std::endl;
        return;
    }
    std::cout << "\nEKI simulation starting - using preloaded meteorological data\n" << std::endl;

    std::cout << Color::BOLD << Color::CYAN
              << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
              << "  RUNNING SIMULATION\n"
              << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
              << Color::RESET << std::endl;

#ifdef DEBUG
    // === NaN check 4: Before simulation start ===
    checkParticleNaN("Before simulation start");
#endif

    //log_first_particle_concentrations(0, 0.0f);

    while(currentTime < time_end){

        stepStart = std::chrono::high_resolution_clock::now();

        currentTime += dt;

        activationRatio = (currentTime) / time_end;
        t0 = (currentTime - static_cast<int>(currentTime/time_interval)*time_interval) / time_interval;

        // EKI mode: Automatic meteorological data selection (past/future pair)
        int meteo_time_interval = Constants::time_interval;
        int past_meteo_index = static_cast<int>(currentTime / meteo_time_interval);
        int future_meteo_index = past_meteo_index + 1;
        
        // Range check and meteorological data update
        if (past_meteo_index >= 0 && past_meteo_index < g_eki_meteo.num_time_steps) {
            // Past meteorological data (d_pressure_past, unis0)
            FlexPres* past_pres_ptr;
            FlexUnis* past_unis_ptr;
            
            cudaMemcpy(&past_pres_ptr, &g_eki_meteo.d_pressure_array[past_meteo_index], 
                       sizeof(FlexPres*), cudaMemcpyDeviceToHost);
            cudaMemcpy(&past_unis_ptr, &g_eki_meteo.d_surface_array[past_meteo_index], 
                       sizeof(FlexUnis*), cudaMemcpyDeviceToHost);
            
            cudaMemcpy(d_pressure_past, past_pres_ptr,
                       g_eki_meteo.pressure_size, cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_surface_past, past_unis_ptr,
                       g_eki_meteo.surface_size, cudaMemcpyDeviceToDevice);

            // DEBUG: Verify GPU meteorological data at first timestep
            if (timestep == 0) {
                FlexPres sample_pres[3];
                // Sample at particle's expected location (source: 37.0°N, 141.0°E)
                // GFS coordinates: lon_idx = (141 + 179) / 0.5 = 640, lat_idx = (37 + 90) / 0.5 = 254
                int test_xidx = 640;  // 141°E
                int test_yidx = 254;  // 37°N
                int test_zidx = 5;    // ~1000m altitude

                int idx0 = test_xidx * dimY_GFS * dimZ_GFS + test_yidx * dimZ_GFS + test_zidx;
                int idx1 = (test_xidx+1) * dimY_GFS * dimZ_GFS + test_yidx * dimZ_GFS + test_zidx;
                int idx2 = test_xidx * dimY_GFS * dimZ_GFS + (test_yidx+1) * dimZ_GFS + test_zidx;

                cudaMemcpy(sample_pres, &d_pressure_past[idx0], sizeof(FlexPres), cudaMemcpyDeviceToHost);
                cudaMemcpy(sample_pres+1, &d_pressure_past[idx1], sizeof(FlexPres), cudaMemcpyDeviceToHost);
                cudaMemcpy(sample_pres+2, &d_pressure_past[idx2], sizeof(FlexPres), cudaMemcpyDeviceToHost);

                extern std::ofstream* g_log_file;
                if (g_log_file && g_log_file->is_open()) {
                    *g_log_file << "[GPU_METEO_VERIFY] Timestep 0 - GPU meteorological data at source location (141°E, 37°N, ~1km):\n";
                    *g_log_file << "  Grid indices: xidx=" << test_xidx << ", yidx=" << test_yidx << ", zidx=" << test_zidx << "\n";
                    *g_log_file << "  Point [" << test_xidx << "," << test_yidx << "," << test_zidx << "]: UU=" << sample_pres[0].UU << " VV=" << sample_pres[0].VV << " WW=" << sample_pres[0].WW << " m/s\n";
                    *g_log_file << "  Point [" << (test_xidx+1) << "," << test_yidx << "," << test_zidx << "]: UU=" << sample_pres[1].UU << " VV=" << sample_pres[1].VV << " WW=" << sample_pres[1].WW << " m/s\n";
                    *g_log_file << "  Point [" << test_xidx << "," << (test_yidx+1) << "," << test_zidx << "]: UU=" << sample_pres[2].UU << " VV=" << sample_pres[2].VV << " WW=" << sample_pres[2].WW << " m/s\n";
                    *g_log_file << std::flush;
                }
            }

            // Update height data as well
            g_height_levels = g_eki_meteo.h_height[past_meteo_index];

#ifdef DEBUG
            // Verify height data at first timestep only
            if (timestep == 0) {
                printf("[DEBUG] Timestep %d: Index %d height data first 5 values: ", timestep, past_meteo_index);
                for (int i = 0; i < std::min(5, (int)g_height_levels.size()); i++) {
                    printf("%.1f ", g_height_levels[i]);
                }
                printf("... %.1f\n", g_height_levels[g_height_levels.size()-1]);
            }
#endif
            
            // Copy height data to GPU memory
            cudaError_t hgt_err = cudaMemcpy(d_height_levels, g_height_levels.data(), sizeof(float) * dimZ_GFS, cudaMemcpyHostToDevice);
            if (hgt_err != cudaSuccess) {
                // Log to file only (collected by Kernel Error Collector for batch reporting)
                extern std::ofstream* g_log_file;
                if (g_log_file && g_log_file->is_open()) {
                    *g_log_file << "[ERROR] Failed to copy height data to GPU: "
                                << cudaGetErrorString(hgt_err) << "\n" << std::flush;
                }
            }
        }

        // Future meteorological data (d_pressure_future, unis1)
        if (future_meteo_index >= 0 && future_meteo_index < g_eki_meteo.num_time_steps) {
            FlexPres* future_pres_ptr;
            FlexUnis* future_unis_ptr;
            
            cudaMemcpy(&future_pres_ptr, &g_eki_meteo.d_pressure_array[future_meteo_index], 
                       sizeof(FlexPres*), cudaMemcpyDeviceToHost);
            cudaMemcpy(&future_unis_ptr, &g_eki_meteo.d_surface_array[future_meteo_index], 
                       sizeof(FlexUnis*), cudaMemcpyDeviceToHost);
            
            cudaMemcpy(d_pressure_future, future_pres_ptr, 
                       g_eki_meteo.pressure_size, cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_surface_future, future_unis_ptr, 
                       g_eki_meteo.surface_size, cudaMemcpyDeviceToDevice);
        } else {
            // Last time period: Set future data same as past data (prevent extrapolation)
            if (past_meteo_index >= 0 && past_meteo_index < g_eki_meteo.num_time_steps) {
                FlexPres* past_pres_ptr;
                FlexUnis* past_unis_ptr;
                
                cudaMemcpy(&past_pres_ptr, &g_eki_meteo.d_pressure_array[past_meteo_index], 
                           sizeof(FlexPres*), cudaMemcpyDeviceToHost);
                cudaMemcpy(&past_unis_ptr, &g_eki_meteo.d_surface_array[past_meteo_index], 
                           sizeof(FlexUnis*), cudaMemcpyDeviceToHost);
                
                cudaMemcpy(d_pressure_future, past_pres_ptr, 
                           g_eki_meteo.pressure_size, cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_surface_future, past_unis_ptr, 
                           g_eki_meteo.surface_size, cudaMemcpyDeviceToDevice);
            }
        }

#ifdef DEBUG
        // === Detailed NaN check at first timestep only ===
        if (timestep == 0) {
            printf("[DEBUG] First timestep - detailed check before kernel execution\n");
            printf("  Current time: %.1fs, t0: %.6f\n", currentTime, t0);
            printf("  Past meteo index: %d, Future meteo index: %d\n", past_meteo_index, future_meteo_index);
            checkParticleNaN("Before first kernel execution", 5);
            checkMeteoDataNaN("After meteorological data update");
        }
#endif

        // Use ensemble kernel for ensemble mode, regular kernel for single mode
        if (is_ensemble_mode) {
            // Ensemble mode: activate particles independently per ensemble
            int total_particles = h_part.size();
            int particles_per_ensemble = nop;  // Particles per ensemble (10000)
            updateParticleFlagsEnsemble<<<blocks, threadsPerBlock>>>
                (d_part, activationRatio, total_particles, particles_per_ensemble);
        } else {
            // Single mode: use standard activation
            updateParticleFlags<<<blocks, threadsPerBlock>>>
                (d_part, activationRatio, nop);
        }
        cudaDeviceSynchronize();
        CHECK_KERNEL_ERROR();

#ifdef DEBUG
        // === NaN check after first kernel ===
        if (timestep == 0) {
            checkParticleNaN("After update_particle_flags", 5);
        }
#endif

        NuclideConfig* nucConfig = NuclideConfig::getInstance();

        // Populate KernelScalars for particle movement kernel
        KernelScalars ks{};
        ks.turb_switch = g_turb_switch;
        ks.drydep = g_drydep;
        ks.wetdep = g_wetdep;
        ks.raddecay = g_raddecay;
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
        ks.T_matrix = d_T_matrix;
        ks.height_levels = d_height_levels;

        // Use ensemble kernel for ensemble mode, regular kernel for single mode
        if (is_ensemble_mode) {
            // Ensemble mode: process all particles (d_nop × ensemble_size)
            int total_particles = h_part.size();
            advectParticlesEnsembleWithVTK<<<blocks, threadsPerBlock>>>
            (d_part, t0, PROCESS_INDEX, d_dryDep, d_wetDep, mesh.lon_count, mesh.lat_count,
                d_surface_past,
                d_pressure_past,
                d_surface_future,
                d_pressure_future,
                total_particles,
                ks);
        } else {
            // Single mode: process d_nop particles
            advectParticlesWithVTK<<<blocks, threadsPerBlock>>>
            (d_part, t0, PROCESS_INDEX, d_dryDep, d_wetDep, mesh.lon_count, mesh.lat_count,
                d_surface_past,
                d_pressure_past,
                d_surface_future,
                d_pressure_future,
                ks);
        }
        cudaDeviceSynchronize();
        CHECK_KERNEL_ERROR();

#ifdef DEBUG
        // === Most important check: After move_part_by_wind_mpi ===
        if (timestep == 0) {
            checkParticleNaN("After move_part_by_wind_mpi (first)", 5);
        } else if (timestep <= 3) {
            checkParticleNaN("After move_part_by_wind_mpi (timestep " + std::to_string(timestep) + ")", 3);
        }
#endif

        timestep++; 

        // Debug: Copy and print first particle position every timestep (first 10 steps)
        if(timestep <= 10) {
            LDMpart first_particle;
            cudaError_t err = cudaMemcpy(&first_particle, d_part, sizeof(LDMpart), cudaMemcpyDeviceToHost);
            if (err == cudaSuccess) {
                // Convert from GFS grid coordinates to geographic coordinates
                float lon = first_particle.x * 0.5f - 179.0f;
                float lat = first_particle.y * 0.5f - 90.0f;
                float z = first_particle.z;

                extern std::ofstream* g_log_file;
                if (g_log_file && g_log_file->is_open()) {
                    *g_log_file << "[PARTICLE_POSITION] Timestep " << timestep << " (t=" << currentTime << "s): "
                                << "lon=" << lon << "°E, lat=" << lat << "°N, z=" << z << "m, "
                                << "flag=" << (first_particle.flag ? "active" : "inactive") << ", "
                                << "timeidx=" << first_particle.timeidx << ", "
                                << "activationRatio=" << activationRatio << ", "
                                << "maxActiveTimeidx=" << int(h_part.size() * activationRatio) << ", "
                                << "u_wind=" << first_particle.u_wind << ", v_wind=" << first_particle.v_wind << ", w_wind=" << first_particle.w_wind << " m/s\n"
                                << std::flush;
                }
            }
        }

        // EKI observation system - check if we should record observations (every timestep)
        if (is_ensemble_mode) {
            // Ensemble mode: compute observations for all ensemble members
            computeReceptorObservations_AllEnsembles(timestep, currentTime, ensemble_size, ensemble_num_states);
        } else {
            // Single mode: compute observations for single simulation
            computeReceptorObservations(timestep, currentTime);
        }

        if(timestep % freq_output==0){
            static bool first_time = true;

            int total_steps = (int)(time_end/dt);
            float progress = (float)timestep / total_steps * 100.0f;
            int bar_width = 40;
            int bar_filled = (int)(progress / 100.0f * bar_width);

            // Terminal: Use ANSI codes for in-place update (goes to both terminal and log via TeeStreambuf)
            if (!first_time && g_sim.fixedScrollOutput) {
                fprintf(stderr, "\033[6A");  // Move cursor up 6 lines to overwrite progress bar
            }

            fprintf(stderr, "\r-------------------------------------------------\033[K\n");
            fprintf(stderr, "\rTime: %8.1f sec │ Step: %4d/%4d [",
                   currentTime, timestep, total_steps);
            for (int i = 0; i < bar_width; i++) {
                if (i < bar_filled) fprintf(stderr, "█");
                else fprintf(stderr, "░");
            }
            fprintf(stderr, "] %.1f%%\033[K\n", progress);
            fprintf(stderr, "\rMeteo: Past=%d Future=%d │ t0=%.3f\033[K\n",
                   past_meteo_index,
                   (future_meteo_index < g_eki_meteo.num_time_steps) ? future_meteo_index : past_meteo_index,
                   t0);

            // Mode information
            if (is_ensemble_mode) {
                fprintf(stderr, "\rMode: ENSEMBLE │ Size: %d\033[K\n",
                        ensemble_size);
            } else {
                fprintf(stderr, "\rMode: SINGLE │ Particles: %d\033[K\n", (int)h_part.size());
            }

            // VTK status - show "Writing files..." if currently outputting VTK
            if (!enable_vtk_output) {
                fprintf(stderr, "\rVTK: Disabled\033[K\n");
            } else if (timestep % freq_output == 0) {
                fprintf(stderr, "\rVTK: Writing files (slow)...\033[K\n");
            } else {
                fprintf(stderr, "\rVTK: Enabled\033[K\n");
            }

            fprintf(stderr, "\r-------------------------------------------------\033[K\n");
            fflush(stderr);

            first_time = false;

            // Use ensemble output for ensemble mode, regular output for single mode
            // VTK output is DISABLED for performance - only enabled for final EKI iteration via enable_vtk_output flag
            if (enable_vtk_output) {
                if (is_ensemble_mode) {
                    outputParticlesBinaryMPI_ens(timestep);
                } else {
                    outputParticlesBinaryMPI(timestep);
                }
            }

            // // Log concentration data for analysis
            // log_first_particle_concentrations(timestep, currentTime);
            // log_all_particles_nuclide_ratios(timestep, currentTime);
            // log_first_particle_cram_detail(timestep, currentTime, dt);
            // log_first_particle_decay_analysis(timestep, currentTime);
            
            // // Export validation reference data
            // exportValidationData(timestep, currentTime);
        }
        stepEnd = std::chrono::high_resolution_clock::now();
        auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(stepEnd - stepStart);
        totalElapsedTime += static_cast<double>(duration0.count()/1.0e6);

        // EKI mode: No meteorological data loading needed (all data already in memory)
        // Previous loadFlexGFSData() call removed

    }

    // Close the progress bar with a final separator line
    fprintf(stderr, "\r-------------------------------------------------\033[K\n");
    fflush(stderr);

    std::cout << "EKI simulation completed" << std::endl;
    std::cout << "Elapsed data time: " << totalElapsedTime << " seconds" << std::endl;
}

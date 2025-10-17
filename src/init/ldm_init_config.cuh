/**
 * @file ldm_init_config.cuh
 * @brief Configuration loading and initialization functions for LDM simulation
 *
 * @details Provides functions for loading simulation parameters, EKI settings,
 *          and initializing grid receptors for debugging modes.
 */

#pragma once

#ifndef LDM_CLASS_DECLARED
#error "Include the header that declares class LDM before including ldm_init_config.cuh"
#endif

/**
 * @method LDM::loadSimulationConfiguration
 * @brief Load simulation parameters from configuration files
 *
 * @details Loads settings from input/setting.txt and source.txt including:
 *          - Time parameters (time_end, dt, output frequency)
 *          - Particle count and physical properties
 *          - Physics model switches (turbulence, deposition, decay)
 *          - Source locations and emission values
 *          - Species properties (decay constants, sizes, densities)
 *
 * @pre Configuration files must exist:
 *      - input/setting.txt (simulation parameters)
 *      - input/source.txt (source locations and emissions)
 *      - cram/A60.csv (CRAM decay matrix, if radioactive decay enabled)
 *
 * @post Global configuration loaded into member variables
 * @post CUDA constant memory initialized with parameters
 * @post Physics model switches copied to GPU
 * @post Output directory cleaned
 * @post CRAM system initialized (if g_raddecay == 1)
 *
 * @algorithm
 *   1. Load setting.txt using ConfigParser
 *   2. Parse physics model switches
 *   3. Clean output directory
 *   4. Load species properties (decay constants, sizes, densities)
 *   5. Initialize CRAM system with dt from config
 *   6. Parse source.txt:
 *      - [SOURCE] section: source coordinates
 *      - [SOURCE_TERM] section: decay constants and deposition velocities
 *      - [RELEASE_CASES] section: emission values per case
 *   7. Upload all parameters to CUDA constant memory
 *
 * @note Physics switches: TURB, DRYDEP, WETDEP, RADDECAY (0=off, 1=on)
 * @note Species arrays support up to 4 species (PROCESS_INDEX indexes them)
 *
 * @throws std::exit(1) if configuration files cannot be opened
 */
void loadSimulationConfiguration();

/**
 * @method LDM::cleanOutputDirectory
 * @brief Remove previous output files before simulation starts
 *
 * @details Cleans output/ directory by removing:
 *          - *.vtk files (VTK particle visualization)
 *          - *.csv files (validation data)
 *          - *.txt files (text output)
 *
 * @post output/ directory cleared of previous run data
 *
 * @note Platform-specific implementation:
 *       - Windows: uses `del /Q output\*.*`
 *       - Linux/macOS: uses `rm -f output/*.vtk output/*.csv output/*.txt`
 *
 * @note Errors suppressed (2>nul on Windows, 2>/dev/null on Unix)
 */
void cleanOutputDirectory();

/**
 * @method LDM::loadReceptorConfig
 * @brief Load receptor configuration from receptor.conf
 *
 * @details Parses input/receptor.conf to configure measurement stations:
 *          - Number of receptors
 *          - Receptor locations (lat/lon pairs)
 *          - Receptor capture radius
 *
 * @pre input/receptor.conf must exist (or falls back to eki_settings.txt)
 * @pre EKI mode must be enabled (g_eki.mode = true)
 *
 * @post g_eki.num_receptors set
 * @post g_eki.receptor_locations populated with coordinates
 * @post g_eki.receptor_capture_radius set
 *
 * @algorithm
 *   1. Try to open receptor.conf, fallback to eki_settings.txt
 *   2. Parse key-value pairs: NUM_RECEPTORS, RECEPTOR_CAPTURE_RADIUS
 *   3. Parse RECEPTOR_LOCATIONS section (multi-line lat/lon pairs)
 *   4. Validate all values (geographic ranges, counts)
 *
 * @throws std::exit(1) if no configuration file can be opened
 * @throws std::exit(1) if validation fails
 *
 * @see input/receptor.conf for format specification
 */
void loadReceptorConfig();

/**
 * @method LDM::loadEKISettings
 * @brief Load EKI-specific configuration from eki.conf
 *
 * @details Parses input/eki.conf to configure Ensemble Kalman Inversion:
 *          - True emission time series (for generating observations)
 *          - Prior emission time series (initial guess)
 *          - EKI algorithm parameters (ensemble size, iterations)
 *          - GPU acceleration settings
 *
 * @pre input/eki.conf must exist (or falls back to eki_settings.txt)
 * @pre EKI mode must be enabled (g_eki.mode = true)
 * @pre loadReceptorConfig() should be called first
 *
 * @post g_eki struct populated with all EKI parameters
 * @post Emission time series stored in g_eki.true_emissions / g_eki.prior_emissions
 *
 * @algorithm
 *   1. Initialize g_eki with default values
 *   2. Parse file line by line:
 *      - Section headers: TRUE_EMISSION_SERIES=, PRIOR_EMISSION_SERIES=
 *      - Key-value pairs: EKI_ENSEMBLE_SIZE=50, EKI_ITERATION=10, etc.
 *      - Matrix data: emission values
 *   3. Handle state machine for multi-line sections
 *   4. Validate all parameters
 *
 * @configuration_keys
 *   - EKI_TIME_INTERVAL: Time step for emission series (float)
 *   - EKI_TIME_UNIT: Time unit (string: "minutes", "hours")
 *   - EKI_ENSEMBLE_SIZE: Number of ensemble members (int)
 *   - EKI_NOISE_LEVEL: Observation noise level (float)
 *   - EKI_ITERATION: Maximum EKI iterations (int)
 *   - EKI_ADAPTIVE: Enable adaptive step size (On/Off)
 *   - EKI_LOCALIZED: Enable covariance localization (On/Off)
 *   - EKI_REGULARIZATION: Enable REnKF regularization (On/Off)
 *   - EKI_GPU_FORWARD/INVERSE: GPU acceleration (On/Off)
 *   - MEMORY_DOCTOR_MODE: Enable IPC debugging (On/Off)
 *
 * @throws std::exit(1) if eki.conf cannot be opened
 * @throws std::exit(1) if validation fails
 */
void loadEKISettings();

/**
 * @method LDM::initializeGridReceptors
 * @brief Initialize uniform grid of receptors for debugging/validation
 *
 * @details Creates a (2N+1)×(2N+1) grid of receptors centered at source location
 *          for detailed spatial analysis of particle dispersion. Used in
 *          receptor-debug mode for validation against analytical solutions.
 *
 * @param[in] grid_count_param Grid extent (N receptors in each direction)
 * @param[in] grid_spacing_param Spacing between receptors (degrees)
 *
 * @pre CUDA device must be initialized
 * @pre Source location should be set (defaults to 37°N, 141°E)
 *
 * @post d_grid_receptor_lats: GPU array of receptor latitudes
 * @post d_grid_receptor_lons: GPU array of receptor longitudes
 * @post d_grid_receptor_dose: GPU dose accumulation array (initialized to 0)
 * @post d_grid_receptor_particle_count: GPU particle count array (initialized to 0)
 * @post grid_receptor_observations: Host storage for observations
 *
 * @algorithm
 *   1. Calculate total receptors = (2*grid_count + 1)²
 *   2. Generate receptor grid centered at source:
 *      lat = source_lat + i * grid_spacing, i ∈ [-N, N]
 *      lon = source_lon + j * grid_spacing, j ∈ [-N, N]
 *   3. Allocate GPU memory for receptor arrays
 *   4. Copy locations to GPU
 *   5. Initialize dose/count arrays to zero
 *
 * @example
 *   initializeGridReceptors(5, 0.1);
 *   // Creates 11×11 = 121 receptors, 0.1° spacing (~11 km)
 *   // Covers area: [36.5-37.5°N, 140.5-141.5°E]
 *
 * @note Typical usage: grid_count=5-10, spacing=0.05-0.2 degrees
 * @note Large grids (>20×20) may impact performance
 * @note Grid is always square and centered at source
 *
 * @memory
 *   GPU: 4 * total_receptors * sizeof(float) + 1 * total_receptors * sizeof(int)
 *   Example: 121 receptors = 2.4 KB
 */
void initializeGridReceptors(int grid_count_param, float grid_spacing_param);

// ================== MODERNIZED CONFIG PARSERS (Phase 1) ==================

/**
 * @method LDM::loadSimulationConfig
 * @brief Load simulation parameters from modernized simulation.conf file
 *
 * @details Parses input/simulation.conf to load core simulation settings:
 *          - Temporal: time_end, time_step, vtk_output_frequency
 *          - Particles: total_particles
 *          - Atmosphere: rural_conditions, use_pasquill_gifford
 *          - Meteorology: use_gfs_data
 *          - Terminal: fixed_scroll_output
 *
 * @pre input/simulation.conf must exist
 * @post g_sim struct populated with simulation parameters
 *
 * @throws std::exit(1) if simulation.conf cannot be opened
 *
 * @see docs/INPUT_MODERNIZATION_PLAN.md
 */
void loadSimulationConfig();

/**
 * @method LDM::loadPhysicsConfig
 * @brief Load physics model configuration from physics.conf
 *
 * @details Parses input/physics.conf to configure physics models and constants.
 *
 * @pre input/physics.conf must exist
 * @post Physics switches stored in g_turb_switch, g_drydep, g_wetdep, g_raddecay
 *
 * @throws std::exit(1) if physics.conf cannot be opened
 */
void loadPhysicsConfig();

/**
 * @method LDM::loadSourceConfig
 * @brief Load source locations from source.conf file
 *
 * @details Loads emission source locations from input/source.conf.
 *          Format: LONGITUDE LATITUDE HEIGHT (space-separated)
 *
 * @pre input/source.conf must exist
 * @post this->sources vector populated with Source structs
 *
 * @throws std::exit(1) if source.conf cannot be opened
 */
void loadSourceConfig();

/**
 * @method LDM::loadNuclidesConfig
 * @brief Load nuclide configuration from nuclides.conf
 *
 * @details Parses nuclide properties with backward compatibility for legacy formats.
 *
 * @pre Nuclide configuration file must exist
 * @post decayConstants and drydepositionVelocity vectors populated
 *
 * @throws std::exit(1) if no configuration file can be opened
 */
void loadNuclidesConfig();

/**
 * @method LDM::loadAdvancedConfig
 * @brief Load advanced system configuration from advanced.conf
 *
 * @details Validates grid dimensions and coordinate system parameters.
 *
 * @pre input/advanced.conf must exist
 * @post Grid dimensions validated against Constants namespace
 *
 * @throws std::exit(1) if advanced.conf cannot be opened
 */
void loadAdvancedConfig();

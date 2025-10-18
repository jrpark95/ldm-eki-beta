/******************************************************************************
 * @file ldm_init_config.cu
 * @brief Configuration loading and initialization for LDM simulation system
 *
 * @details Implements configuration parsers for all input files:
 *          - setting.txt: Core simulation parameters
 *          - source.txt: Emission source locations and release cases
 *          - eki_settings.txt: Ensemble Kalman Inversion parameters
 *          - Modernized config files: simulation.conf, physics.conf, etc.
 *
 * @note Legacy file support maintained for backward compatibility
 * @note New modular config system introduced in 2025-10-17 (Phase 1)
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/

#include "../core/ldm.cuh"
#include "../physics/ldm_nuclides.cuh"
#include "colors.h"

/******************************************************************************
 * @brief Load simulation configuration from legacy setting.txt file
 *
 * @details Parses input/setting.txt to load core simulation parameters and
 *          source.txt for emission source definitions. This is the legacy
 *          configuration loader maintained for backward compatibility.
 *
 *          Configuration includes:
 *          - Temporal parameters: time_end, dt, output frequency
 *          - Particle properties: count, size distribution, density
 *          - Physics model switches: turbulence, deposition, decay
 *          - Atmospheric conditions: rural/urban, stability parameterization
 *          - Meteorological data source: GFS/LDAPS selection
 *          - Source locations: coordinates (lon, lat, height)
 *          - Release cases: emission values per source/time
 *
 * @pre Input files must exist:
 *      - input/setting.txt (simulation parameters)
 *      - input/source.txt (source locations and release cases)
 *      - cram/A60.csv (CRAM decay matrix if radioactive decay enabled)
 *
 * @post Member variables populated:
 *       - time_end, dt, freq_output, nop
 *       - isRural, isPG, isGFS
 *       - sources vector, concentrations vector
 *       - decayConstants, drydepositionVelocity vectors
 * @post Physics model switches set: g_turb_switch, g_drydep, g_wetdep, g_raddecay
 * @post CRAM system initialized (if radioactive decay enabled)
 * @post Output directory cleaned
 *
 * @algorithm
 *   1. Load setting.txt using ConfigReader
 *   2. Parse simulation parameters (time, particle count, etc.)
 *   3. Load physics model switches
 *   4. Parse species properties (decay constants, sizes, densities)
 *   5. Initialize CRAM decay system with dt from config
 *   6. Open source.txt for emission source parsing
 *   7. Parse [SOURCE] section: lon, lat, height coordinates
 *   8. Parse [SOURCE_TERM] section: decay constants, deposition velocities
 *   9. Parse [RELEASE_CASES] section: location, source term, emission value
 *  10. Close file and clean output directory
 *
 * @note Configuration values passed to GPU kernels via KernelScalars struct
 * @note No longer uses __constant__ memory (removed in refactoring)
 *
 * @see loadSimulationConfig() for modernized config file loader
 * @see input/setting.txt for legacy file format specification
 * @see input/source.txt for source definition format
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::loadSimulationConfiguration(){

    if (!g_config.loadConfig("input/setting.txt")) {
        std::cerr << "Failed to load configuration file" << std::endl;
        exit(1);
    }

    FILE* sourceFile;

    // Parse temporal parameters
    time_end = g_config.getFloat("Time_end(s)", 64800.0f);      // Simulation duration (seconds)
    dt = g_config.getFloat("dt(s)", 10.0f);                     // Time step (seconds)
    freq_output = g_config.getInt("Plot_output_freq", 10);      // VTK output frequency

    // Parse particle parameters
    nop = g_config.getInt("Total_number_of_particle", 10000);   // Total particle count

    // Hardcode atmospheric conditions and meteorological data (v1.0 production settings)
    isRural = 1;  // HARDCODED: Rural conditions
    isPG = 1;     // HARDCODED: Pasquill-Gifford stability scheme
    isGFS = 1;    // HARDCODED: GFS meteorological data

    // Load terminal output settings
    g_sim.fixedScrollOutput = g_config.getInt("fixed_scroll_output", 1);

    // Hardcode turbulence model (not implemented in v1.0)
    g_turb_switch = 0;  // HARDCODED: Turbulence model not implemented

    // Note: Other physics models (dry/wet deposition, radioactive decay)
    // are loaded from physics.conf in loadPhysicsConfig()

    // Clean output directory before simulation
    cleanOutputDirectory();

    // Parse species properties (up to 4 species supported)
    std::vector<std::string> species_names = g_config.getStringArray("species_names");
    std::vector<float> decay_constants = g_config.getFloatArray("decay_constants");
    std::vector<float> deposition_velocities = g_config.getFloatArray("deposition_velocities");
    std::vector<float> particle_sizes = g_config.getFloatArray("particle_sizes");
    std::vector<float> particle_densities = g_config.getFloatArray("particle_densities");
    std::vector<float> size_standard_deviations = g_config.getFloatArray("size_standard_deviations");

    for (int i = 0; i < 4 && i < species_names.size(); i++) {
        g_mpi.species[i] = species_names[i];
        g_mpi.decayConstants[i] = (i < decay_constants.size()) ? decay_constants[i] : 1.00e-6f;
        g_mpi.depositionVelocities[i] = (i < deposition_velocities.size()) ? deposition_velocities[i] : 0.01f;
        g_mpi.particleSizes[i] = (i < particle_sizes.size()) ? particle_sizes[i] : 0.6f;
        g_mpi.particleDensities[i] = (i < particle_densities.size()) ? particle_densities[i] : 2500.0f;
        g_mpi.sizeStandardDeviations[i] = (i < size_standard_deviations.size()) ? size_standard_deviations[i] : 0.01f;
    }

    // Initialize CRAM decay system with dynamic dt from configuration
    if (initialize_cram_system("cram/A60.csv")) {
        // Successfully computed exp(-A*dt) matrix for CRAM decay
    } else {
        std::cerr << "Warning: CRAM system initialization failed, using traditional decay" << std::endl;
    }

    // Open source configuration file
    std::string source_file_path = g_config.getString("input_base_path", "./input/") + "source.txt";
    sourceFile = fopen(source_file_path.c_str(), "r");

    if (!sourceFile){
        std::cerr << "Failed to open source.txt" << std::endl;
        exit(1);
    }

    // Parse source.txt with three sections: [SOURCE], [SOURCE_TERM], [RELEASE_CASES]
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), sourceFile)){
        if (buffer[0] == '#') continue;  // Skip comments

        // Parse [SOURCE] section: source coordinates (lon, lat, height)
        if (strstr(buffer, "[SOURCE]")) {
            while (fgets(buffer, sizeof(buffer), sourceFile) && !strstr(buffer, "[SOURCE_TERM]")) {
                if (buffer[0] == '#') continue;

                Source src;
                sscanf(buffer, "%f %f %f", &src.lon, &src.lat, &src.height);
                sources.push_back(src);
            }
            sources.pop_back();  // Remove sentinel entry
        }

        // Parse [SOURCE_TERM] section: decay constants and deposition velocities
        if (strstr(buffer, "[SOURCE_TERM]")){
            while (fgets(buffer, sizeof(buffer), sourceFile) && !strstr(buffer, "[RELEASE_CASES]")) {
                if (buffer[0] == '#') continue;

                int srcnum;
                float decay, depvel;
                sscanf(buffer, "%d %f %f", &srcnum, &decay, &depvel);
                decayConstants.push_back(decay);
                drydepositionVelocity.push_back(depvel);
            }
            decayConstants.pop_back();      // Remove sentinel entry
            drydepositionVelocity.pop_back();
        }

        // Parse [RELEASE_CASES] section: emission scenarios
        if (strstr(buffer, "[RELEASE_CASES]")){
            while (fgets(buffer, sizeof(buffer), sourceFile)) {
                if (buffer[0] == '#') continue;
                Concentration conc;
                sscanf(buffer, "%d %d %lf", &conc.location, &conc.sourceterm, &conc.value);
                concentrations.push_back(conc);
            }
        }
    }

    fclose(sourceFile);

    // Note: Configuration values now passed via KernelScalars struct to kernels
    // __constant__ memory symbols removed during 2025 refactoring (non-RDC compatibility)

}
/******************************************************************************
 * @brief Clean output directory before simulation starts
 *
 * @details Removes previous run artifacts from output/ directory to prevent
 *          data contamination between simulation runs. Platform-specific
 *          implementation using system calls.
 *
 * @post output/ directory cleared of:
 *       - *.vtk files (VTK particle visualization)
 *       - *.csv files (validation data)
 *       - *.txt files (text output)
 *
 * @note Platform-specific behavior:
 *       - Windows: Uses 'del /Q output\*.*' command
 *       - Linux/macOS: Uses 'rm -f output/*.{vtk,csv,txt}' commands
 * @note Errors suppressed (2>nul on Windows, 2>/dev/null on Unix)
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::cleanOutputDirectory() {
    std::cout << "Cleaning output directory... " << std::flush;

    // Remove all output files using platform-specific commands
    #ifdef _WIN32
        system("del /Q output\\*.* 2>nul");
    #else
        system("rm -f output/*.vtk 2>/dev/null");
        system("rm -f output/*.csv 2>/dev/null");
        system("rm -f output/*.txt 2>/dev/null");
    #endif

    std::cout << Color::GREEN << "" << Color::RESET << std::endl;
}

/******************************************************************************
 * @brief Load Ensemble Kalman Inversion settings from eki_settings.txt
 *
 * @details Parses input/eki_settings.txt to configure the EKI optimization
 *          framework. Uses a state machine to parse multi-line sections
 *          (receptor locations, emission time series) and key-value pairs.
 *
 *          Configuration includes:
 *          - Receptor definitions: locations (lat/lon), capture radius
 *          - Emission time series: true values (for observations), prior guess
 *          - EKI algorithm parameters: ensemble size, iterations, noise level
 *          - Algorithm variants: adaptive step size, localization, regularization
 *          - GPU acceleration settings: forward/inverse model GPU usage
 *          - Debug options: Memory Doctor mode for IPC diagnostics
 *
 * @pre input/eki_settings.txt must exist
 * @pre EKI mode must be enabled (function called when g_eki.mode = true)
 *
 * @post g_eki struct fully populated with EKI parameters
 * @post g_eki.receptor_locations: vector of (lat, lon) pairs
 * @post g_eki.true_emissions: time series for generating observations
 * @post g_eki.prior_emissions: initial guess for optimization
 * @post Algorithm switches set: adaptive_eki, localized_eki, regularization
 *
 * @algorithm State machine parser:
 *   1. Initialize g_eki with default values
 *   2. Parse file line by line:
 *      - Section headers toggle state flags:
 *        * RECEPTOR_LOCATIONS_MATRIX= → read receptor coordinates
 *        * TRUE_EMISSION_SERIES= → read true emission values
 *        * PRIOR_EMISSION_SERIES= → read prior emission values
 *      - Key-value pairs (KEY=VALUE) reset state flags and parse parameters
 *      - Matrix data lines parsed according to current state
 *   3. Validate configuration (e.g., num_receptors matches location count)
 *   4. Print essential EKI configuration summary
 *
 * @note File format:
 *       - Comments: Lines starting with #
 *       - Key-value: KEY=VALUE (no spaces around =)
 *       - Matrix sections: Header line followed by data lines
 * @note State machine ensures correct parsing of multi-line sections
 *
 * @see input/eki_settings.txt for configuration file format
 * @see src/eki/RunEstimator.py for Python EKI executor
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::loadReceptorConfig() {
    // Try modern config file first, fallback to legacy (embedded in eki_settings.txt)
    FILE* receptorFile = fopen("input/receptor.conf", "r");
    const char* config_filename = "input/receptor.conf";

    if (!receptorFile) {
        receptorFile = fopen("input/eki_settings.txt", "r");
        config_filename = "input/eki_settings.txt";
    }

    if (!receptorFile) {
        std::cerr << "\n" << Color::RED << Color::BOLD << "[ERROR] " << Color::RESET
                  << "Failed to open receptor configuration file" << std::endl;
        std::cerr << "  Tried: input/receptor.conf, input/eki_settings.txt" << std::endl;
        exit(1);
    }

    std::cout << "\n" << Color::CYAN << "[SYSTEM] " << Color::RESET
              << "Loading receptor settings from " << Color::BOLD << config_filename << Color::RESET << "... " << std::flush;

    char buffer[256];

    // Initialize receptor configuration
    g_eki.num_receptors = 0;
    g_eki.receptor_locations.clear();

    // State machine flag for multi-line section parsing
    bool reading_receptor_locations = false;

    while (fgets(buffer, sizeof(buffer), receptorFile)) {
        // Skip comments and empty lines
        if (buffer[0] == '#' || buffer[0] == '\n' || buffer[0] == '\r') {
            continue;
        }

        // Normalize separator: convert ':' to '=' for uniform parsing
        char* colon_pos = strchr(buffer, ':');
        if (colon_pos && !strchr(buffer, '=')) {
            *colon_pos = '=';
        }

        // Check for multi-line section header
        if (strstr(buffer, "RECEPTOR_LOCATIONS=") || strstr(buffer, "RECEPTOR_LOCATIONS_MATRIX=")) {
            reading_receptor_locations = true;
            continue;
        }

        // Reset section flag when encountering key-value pairs
        if (strchr(buffer, '=') != nullptr) {
            reading_receptor_locations = false;
        }

        // Parse receptor location data
        if (reading_receptor_locations) {
            float lat, lon;
            if (sscanf(buffer, "%f %f", &lat, &lon) == 2) {
                g_eki.receptor_locations.push_back(std::make_pair(lat, lon));
            }
        }

        // Parse key-value pairs
        if (strchr(buffer, '=') != nullptr) {
            if (strstr(buffer, "NUM_RECEPTORS=")) {
                sscanf(buffer, "NUM_RECEPTORS=%d", &g_eki.num_receptors);
            }
            else if (strstr(buffer, "RECEPTOR_CAPTURE_RADIUS=")) {
                sscanf(buffer, "RECEPTOR_CAPTURE_RADIUS=%f", &g_eki.receptor_capture_radius);
            }
        }
    }

    fclose(receptorFile);

    // ========== COMPREHENSIVE VALIDATION ==========

    // ===== VALIDATION: NUM_RECEPTORS =====
    if (g_eki.num_receptors <= 0) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Invalid NUM_RECEPTORS: " << g_eki.num_receptors << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    At least one receptor must be defined for EKI mode." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required value:" << Color::RESET << std::endl;
        std::cerr << "    NUM_RECEPTORS >= 1" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Recommended:" << Color::RESET << " 3-10 receptors for good spatial coverage" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " " << config_filename << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    if (g_eki.receptor_locations.size() != static_cast<size_t>(g_eki.num_receptors)) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Receptor count mismatch" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    NUM_RECEPTORS=" << g_eki.num_receptors
                  << " but " << g_eki.receptor_locations.size()
                  << " receptor locations defined" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Solution:" << Color::RESET << std::endl;
        std::cerr << "    Ensure RECEPTOR_LOCATIONS has exactly "
                  << g_eki.num_receptors << " lines" << std::endl;
        std::cerr << "    Format: latitude longitude (one per line)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " " << config_filename << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // ===== VALIDATION: Receptor locations =====
    for (size_t i = 0; i < g_eki.receptor_locations.size(); i++) {
        float lat = g_eki.receptor_locations[i].first;
        float lon = g_eki.receptor_locations[i].second;

        if (lat < -90.0f || lat > 90.0f) {
            std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                      << Color::RESET << "Invalid receptor latitude: " << lat << "° (receptor "
                      << (i+1) << ")" << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::CYAN << "Required range:" << Color::RESET << std::endl;
            std::cerr << "    -90.0 <= latitude <= 90.0 (degrees)" << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET
                      << " " << config_filename << ", RECEPTOR_LOCATIONS" << std::endl;
            std::cerr << std::endl;
            exit(1);
        }

        if (lon < -180.0f || lon > 180.0f) {
            std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                      << Color::RESET << "Invalid receptor longitude: " << lon << "° (receptor "
                      << (i+1) << ")" << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::CYAN << "Required range:" << Color::RESET << std::endl;
            std::cerr << "    -180.0 <= longitude <= 180.0 (degrees)" << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET
                      << " " << config_filename << ", RECEPTOR_LOCATIONS" << std::endl;
            std::cerr << std::endl;
            exit(1);
        }
    }

    // ===== VALIDATION: RECEPTOR_CAPTURE_RADIUS =====
    if (g_eki.receptor_capture_radius <= 0.0f) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Invalid RECEPTOR_CAPTURE_RADIUS: "
                  << g_eki.receptor_capture_radius << "°" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Capture radius must be positive." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required value:" << Color::RESET << std::endl;
        std::cerr << "    RECEPTOR_CAPTURE_RADIUS > 0.0 (degrees)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Typical values:" << Color::RESET << std::endl;
        std::cerr << "    - Fine resolution:   0.01° (~1 km)" << std::endl;
        std::cerr << "    - Standard:          0.025° (~2.5 km)" << std::endl;
        std::cerr << "    - Coarse:            0.05° (~5 km)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " " << config_filename << std::endl;
        std::cerr << std::endl;
        exit(1);
    }
    if (g_eki.receptor_capture_radius > 1.0f) {
        std::cerr << std::endl << Color::YELLOW << Color::BOLD << "[WARNING] "
                  << Color::RESET << "Very large RECEPTOR_CAPTURE_RADIUS: "
                  << g_eki.receptor_capture_radius << "° (~"
                  << (g_eki.receptor_capture_radius * 111.0f) << " km)" << std::endl;
        std::cerr << "  This may capture particles from large areas, reducing spatial resolution." << std::endl;
        std::cerr << "  Consider using smaller radius (0.01-0.05°) for better accuracy." << std::endl;
        std::cerr << std::endl;
    }

    std::cout << "done\n";
}

void LDM::loadEKISettings() {
    // Try modern config file first, fallback to legacy
    FILE* ekiFile = fopen("input/eki.conf", "r");
    const char* config_filename = "input/eki.conf";

    if (!ekiFile) {
        ekiFile = fopen("input/eki_settings.txt", "r");
        config_filename = "input/eki_settings.txt";
    }

    if (!ekiFile) {
        std::cerr << "\n" << Color::RED << Color::BOLD << "[ERROR] " << Color::RESET
                  << "Failed to open EKI configuration file" << std::endl;
        std::cerr << "  Tried: input/eki.conf, input/eki_settings.txt" << std::endl;
        exit(1);
    }

    std::cout << "\n" << Color::CYAN << "[SYSTEM] " << Color::RESET
              << "Loading EKI settings from " << Color::BOLD << config_filename << Color::RESET << "... " << std::flush;

    char buffer[256];

    // Initialize global EKI configuration - NO DEFAULTS, all values must be in config file
    g_eki.mode = true;  // EKI mode is enabled by calling this function

    // Clear emission vectors
    g_eki.true_emissions.clear();
    g_eki.prior_emissions.clear();

    // Initialize to sentinel values to detect missing parameters
    g_eki.time_interval = -1.0f;          // Must be set by config
    g_eki.time_unit = "";                 // Must be set by config
    g_eki.prior_mode = "";                // Must be set by config
    g_eki.prior_constant = -1.0f;         // Must be set by config (if prior_mode = constant)
    g_eki.ensemble_size = -1;             // Must be set by config
    g_eki.noise_level = -1.0f;            // Must be set by config
    g_eki.iteration = -1;                 // Must be set by config
    g_eki.perturb_option = "";            // Must be set by config
    g_eki.adaptive_eki = "";              // Must be set by config
    g_eki.localized_eki = "";             // Must be set by config
    g_eki.regularization = "";            // Must be set by config
    g_eki.renkf_lambda = -1.0f;           // Must be set by config (if regularization = on)

    // State machine flags for multi-line section parsing
    bool reading_true_emissions = false;
    bool reading_prior_emissions = false;
    
    while (fgets(buffer, sizeof(buffer), ekiFile)) {
        // Skip comments and empty lines
        if (buffer[0] == '#' || buffer[0] == '\n' || buffer[0] == '\r') {
            continue;
        }

        // Normalize separator: convert ':' to '=' for uniform parsing
        // This allows both "KEY: value" (new format) and "KEY=value" (legacy format)
        char* colon_pos = strchr(buffer, ':');
        if (colon_pos && !strchr(buffer, '=')) {
            *colon_pos = '=';
        }

        // State machine: Check for multi-line section headers
        if (strstr(buffer, "TRUE_EMISSION_SERIES=")) {
            reading_true_emissions = true;
            reading_prior_emissions = false;
            continue;
        }

        if (strstr(buffer, "PRIOR_EMISSION_SERIES=")) {
            reading_true_emissions = false;
            reading_prior_emissions = true;
            continue;
        }

        // Reset section flags when encountering key-value pairs
        if (strchr(buffer, '=') != nullptr) {
            reading_true_emissions = false;
            reading_prior_emissions = false;
        }

        // Parse matrix data based on current state
        if (reading_true_emissions) {
            float emission;
            if (sscanf(buffer, "%f", &emission) == 1) {
                g_eki.true_emissions.push_back(emission);
            }
        }
        else if (reading_prior_emissions) {
            float emission;
            if (sscanf(buffer, "%f", &emission) == 1) {
                g_eki.prior_emissions.push_back(emission);
            }
        }
        
        // Parse key-value pairs (section flags already reset above)
        if (strchr(buffer, '=') != nullptr) {

            // Temporal parameters
            if (strstr(buffer, "EKI_TIME_INTERVAL=")) {
                sscanf(buffer, "EKI_TIME_INTERVAL=%f", &g_eki.time_interval);
            }
            else if (strstr(buffer, "EKI_TIME_UNIT=")) {
                char unit[32];
                sscanf(buffer, "EKI_TIME_UNIT=%s", unit);
                g_eki.time_unit = std::string(unit);
            }

            // Prior emission settings
            else if (strstr(buffer, "PRIOR_EMISSION_MODE=")) {
                char mode[32];
                sscanf(buffer, "PRIOR_EMISSION_MODE=%s", mode);
                g_eki.prior_mode = std::string(mode);
            }
            else if (strstr(buffer, "PRIOR_EMISSION_CONSTANT=")) {
                sscanf(buffer, "PRIOR_EMISSION_CONSTANT=%f", &g_eki.prior_constant);
            }

            // EKI algorithm parameters
            else if (strstr(buffer, "EKI_ENSEMBLE_SIZE=")) {
                sscanf(buffer, "EKI_ENSEMBLE_SIZE=%d", &g_eki.ensemble_size);
            }
            else if (strstr(buffer, "EKI_NOISE_LEVEL=")) {
                sscanf(buffer, "EKI_NOISE_LEVEL=%f", &g_eki.noise_level);
            }
            else if (strstr(buffer, "EKI_ITERATION=")) {
                sscanf(buffer, "EKI_ITERATION=%d", &g_eki.iteration);
            }
            else if (strstr(buffer, "EKI_PERTURB_OPTION=")) {
                char opt[32];
                sscanf(buffer, "EKI_PERTURB_OPTION=%s", opt);
                g_eki.perturb_option = std::string(opt);
            }

            // EKI algorithm variants
            else if (strstr(buffer, "EKI_ADAPTIVE=")) {
                char opt[32];
                sscanf(buffer, "EKI_ADAPTIVE=%s", opt);
                g_eki.adaptive_eki = std::string(opt);
            }
            else if (strstr(buffer, "EKI_LOCALIZED=")) {
                char opt[32];
                sscanf(buffer, "EKI_LOCALIZED=%s", opt);
                g_eki.localized_eki = std::string(opt);
            }
            else if (strstr(buffer, "EKI_REGULARIZATION=")) {
                char opt[32];
                sscanf(buffer, "EKI_REGULARIZATION=%s", opt);
                g_eki.regularization = std::string(opt);
            }
            else if (strstr(buffer, "EKI_RENKF_LAMBDA=")) {
                sscanf(buffer, "EKI_RENKF_LAMBDA=%f", &g_eki.renkf_lambda);
            }

            // Debug mode
            else if (strstr(buffer, "MEMORY_DOCTOR_MODE=")) {
                char mode[32];
                sscanf(buffer, "MEMORY_DOCTOR_MODE=%s", mode);
                g_eki.memory_doctor_mode = (strcmp(mode, "On") == 0 || strcmp(mode, "on") == 0 ||
                                           strcmp(mode, "ON") == 0 || strcmp(mode, "1") == 0);
            }
        }
    }
    
    fclose(ekiFile);

    // ========== COMPREHENSIVE VALIDATION ==========

    // ===== VALIDATION: EKI_TIME_UNIT (must be set) =====
    if (g_eki.time_unit.empty()) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Missing required parameter: EKI_TIME_UNIT" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Time unit for emission time series must be specified." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required value:" << Color::RESET << std::endl;
        std::cerr << "    EKI_TIME_UNIT: <string>" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Valid values:" << Color::RESET << std::endl;
        std::cerr << "    - \"seconds\"" << std::endl;
        std::cerr << "    - \"minutes\"" << std::endl;
        std::cerr << "    - \"hours\"" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Example:" << Color::RESET << std::endl;
        std::cerr << "    EKI_TIME_UNIT: minutes" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " " << config_filename << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // ===== VALIDATION: PRIOR_EMISSION_MODE (must be set) =====
    if (g_eki.prior_mode.empty()) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Missing required parameter: PRIOR_EMISSION_MODE" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Prior emission mode must be specified for EKI." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required value:" << Color::RESET << std::endl;
        std::cerr << "    PRIOR_EMISSION_MODE: <string>" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Valid values:" << Color::RESET << std::endl;
        std::cerr << "    - \"constant\" (single value for all timesteps)" << std::endl;
        std::cerr << "    - \"series\" (use PRIOR_EMISSION_SERIES)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Example:" << Color::RESET << std::endl;
        std::cerr << "    PRIOR_EMISSION_MODE: constant" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " " << config_filename << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // ===== VALIDATION: PRIOR_EMISSION_CONSTANT (if mode = constant) =====
    if (g_eki.prior_mode == "constant" && g_eki.prior_constant < 0.0f) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Missing required parameter: PRIOR_EMISSION_CONSTANT" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    PRIOR_EMISSION_MODE is set to 'constant' but" << std::endl;
        std::cerr << "    PRIOR_EMISSION_CONSTANT is not specified." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required value:" << Color::RESET << std::endl;
        std::cerr << "    PRIOR_EMISSION_CONSTANT: <positive number> (Bq/s)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Typical values:" << Color::RESET << std::endl;
        std::cerr << "    - Small source:  1.0e+8 Bq/s" << std::endl;
        std::cerr << "    - Medium source: 1.0e+10 Bq/s" << std::endl;
        std::cerr << "    - Large source:  1.0e+12 Bq/s" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Example:" << Color::RESET << std::endl;
        std::cerr << "    PRIOR_EMISSION_CONSTANT: 1.5e+8" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " " << config_filename << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // ===== VALIDATION: EKI_PERTURB_OPTION (must be set) =====
    if (g_eki.perturb_option.empty()) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Missing required parameter: EKI_PERTURB_OPTION" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Perturbation option for ensemble generation must be specified." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required value:" << Color::RESET << std::endl;
        std::cerr << "    EKI_PERTURB_OPTION: <string>" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Valid values:" << Color::RESET << std::endl;
        std::cerr << "    - \"observations\" (perturb observations - recommended)" << std::endl;
        std::cerr << "    - \"parameters\" (perturb state parameters)" << std::endl;
        std::cerr << "    - \"both\" (perturb both)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Recommended:" << Color::RESET << " observations" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Example:" << Color::RESET << std::endl;
        std::cerr << "    EKI_PERTURB_OPTION: observations" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " " << config_filename << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // ===== VALIDATION: EKI_ADAPTIVE (must be set) =====
    if (g_eki.adaptive_eki.empty()) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Missing required parameter: EKI_ADAPTIVE" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Adaptive step size option must be specified." << std::endl;
        std::cerr << "    This controls whether the EKI algorithm uses adaptive step sizing." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required value:" << Color::RESET << std::endl;
        std::cerr << "    EKI_ADAPTIVE: <On|Off>" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Recommended:" << Color::RESET << " On (improves convergence)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Example:" << Color::RESET << std::endl;
        std::cerr << "    EKI_ADAPTIVE: On" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " " << config_filename << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // ===== VALIDATION: EKI_LOCALIZED (must be set) =====
    if (g_eki.localized_eki.empty()) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Missing required parameter: EKI_LOCALIZED" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Covariance localization option must be specified." << std::endl;
        std::cerr << "    This controls whether to remove spurious correlations in ensemble." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required value:" << Color::RESET << std::endl;
        std::cerr << "    EKI_LOCALIZED: <On|Off>" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Recommended:" << Color::RESET << " On (for ensemble_size < 100)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Example:" << Color::RESET << std::endl;
        std::cerr << "    EKI_LOCALIZED: On" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " " << config_filename << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // ===== VALIDATION: EKI_REGULARIZATION (must be set) =====
    if (g_eki.regularization.empty()) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Missing required parameter: EKI_REGULARIZATION" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Regularization option must be specified." << std::endl;
        std::cerr << "    This controls whether to use Tikhonov regularization." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required value:" << Color::RESET << std::endl;
        std::cerr << "    EKI_REGULARIZATION: <On|Off>" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Recommended:" << Color::RESET << " Off (unless ill-posed problem)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Example:" << Color::RESET << std::endl;
        std::cerr << "    EKI_REGULARIZATION: Off" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " " << config_filename << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // ===== VALIDATION: EKI_RENKF_LAMBDA (if regularization = On) =====
    if ((g_eki.regularization == "On" || g_eki.regularization == "on") && g_eki.renkf_lambda < 0.0f) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Missing required parameter: EKI_RENKF_LAMBDA" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    EKI_REGULARIZATION is set to 'On' but" << std::endl;
        std::cerr << "    EKI_RENKF_LAMBDA (regularization parameter) is not specified." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required value:" << Color::RESET << std::endl;
        std::cerr << "    EKI_RENKF_LAMBDA: <positive number>" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Typical values:" << Color::RESET << std::endl;
        std::cerr << "    - Weak regularization:   0.01-0.1" << std::endl;
        std::cerr << "    - Medium regularization: 0.1-1.0" << std::endl;
        std::cerr << "    - Strong regularization: 1.0-10.0" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Example:" << Color::RESET << std::endl;
        std::cerr << "    EKI_RENKF_LAMBDA: 0.1" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " " << config_filename << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // ===== VALIDATION: EKI_ENSEMBLE_SIZE =====
    if (g_eki.ensemble_size <= 0) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Invalid EKI_ENSEMBLE_SIZE: " << g_eki.ensemble_size << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Ensemble size must be positive." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required value:" << Color::RESET << std::endl;
        std::cerr << "    EKI_ENSEMBLE_SIZE >= 10 (minimum for Kalman filter)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Recommended range:" << Color::RESET << std::endl;
        std::cerr << "    - Quick test:  20-50 members" << std::endl;
        std::cerr << "    - Standard:    50-100 members (good balance)" << std::endl;
        std::cerr << "    - High quality: 100-500 members" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " " << config_filename << std::endl;
        std::cerr << std::endl;
        exit(1);
    }
    if (g_eki.ensemble_size < 10) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Too few ensemble members: " << g_eki.ensemble_size << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Ensemble Kalman methods require sufficient members to estimate" << std::endl;
        std::cerr << "    covariance matrices. < 10 members produces unreliable results." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required value:" << Color::RESET << std::endl;
        std::cerr << "    EKI_ENSEMBLE_SIZE >= 10" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Recommended:" << Color::RESET << " At least 50 members" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " " << config_filename << std::endl;
        std::cerr << std::endl;
        exit(1);
    }
    if (g_eki.ensemble_size > 10000) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Excessive ensemble size: " << g_eki.ensemble_size << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    This will cause:" << std::endl;
        std::cerr << "    - Enormous memory consumption" << std::endl;
        std::cerr << "    - Extremely long computation times" << std::endl;
        std::cerr << "    - Diminishing returns in accuracy" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required value:" << Color::RESET << std::endl;
        std::cerr << "    EKI_ENSEMBLE_SIZE <= 10000" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Practical maximum:" << Color::RESET << " 500 members" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " " << config_filename << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // ===== VALIDATION: EKI_ITERATION =====
    if (g_eki.iteration <= 0) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Invalid EKI_ITERATION: " << g_eki.iteration << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    At least one iteration is required." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required value:" << Color::RESET << std::endl;
        std::cerr << "    EKI_ITERATION >= 1" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Typical values:" << Color::RESET << std::endl;
        std::cerr << "    - Quick test:  1-3 iterations" << std::endl;
        std::cerr << "    - Standard:    3-5 iterations" << std::endl;
        std::cerr << "    - Convergence: 5-10 iterations" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " " << config_filename << std::endl;
        std::cerr << std::endl;
        exit(1);
    }
    if (g_eki.iteration > 100) {
        std::cerr << std::endl << Color::YELLOW << Color::BOLD << "[WARNING] "
                  << Color::RESET << "Very many iterations: " << g_eki.iteration << std::endl;
        std::cerr << "  This will require extremely long computation time." << std::endl;
        std::cerr << "  Consider using fewer iterations (e.g., 3-10) with convergence checking." << std::endl;
        std::cerr << std::endl;
    }

    // ===== VALIDATION: EKI_NOISE_LEVEL =====
    if (g_eki.noise_level < 0.0f) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Invalid EKI_NOISE_LEVEL: " << g_eki.noise_level << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Noise level cannot be negative." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required value:" << Color::RESET << std::endl;
        std::cerr << "    EKI_NOISE_LEVEL >= 0.0 (fraction, e.g., 0.1 = 10%)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Typical values:" << Color::RESET << std::endl;
        std::cerr << "    - Low noise:     0.01-0.05 (1-5%)" << std::endl;
        std::cerr << "    - Medium noise:  0.05-0.10 (5-10%)" << std::endl;
        std::cerr << "    - High noise:    0.10-0.20 (10-20%)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " " << config_filename << std::endl;
        std::cerr << std::endl;
        exit(1);
    }
    if (g_eki.noise_level > 1.0f) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Excessive EKI_NOISE_LEVEL: " << g_eki.noise_level
                  << " (" << (g_eki.noise_level * 100.0f) << "%)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Noise level > 100% indicates measurement error exceeds signal." << std::endl;
        std::cerr << "    This makes inverse problem ill-posed." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required value:" << Color::RESET << std::endl;
        std::cerr << "    EKI_NOISE_LEVEL <= 1.0 (100%)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Recommended:" << Color::RESET << " 0.05-0.15 (5-15%)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " " << config_filename << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // ===== VALIDATION: TRUE_EMISSION_SERIES =====
    if (g_eki.true_emissions.empty()) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "No TRUE_EMISSION_SERIES data found" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    TRUE_EMISSION_SERIES must have at least one time step." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Solution:" << Color::RESET << std::endl;
        std::cerr << "    Define emission values in TRUE_EMISSION_SERIES section" << std::endl;
        std::cerr << "    One value per line (in Bq)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Example:" << Color::RESET << std::endl;
        std::cerr << "    TRUE_EMISSION_SERIES=" << std::endl;
        std::cerr << "    1.0e+12" << std::endl;
        std::cerr << "    1.0e+12" << std::endl;
        std::cerr << "    5.0e+11" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " " << config_filename << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // Check for negative or unrealistic emission values
    for (size_t i = 0; i < g_eki.true_emissions.size(); i++) {
        if (g_eki.true_emissions[i] < 0.0f) {
            std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                      << Color::RESET << "Negative emission value at timestep " << (i+1)
                      << ": " << g_eki.true_emissions[i] << " Bq" << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
            std::cerr << "    Emission rates cannot be negative." << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET
                      << " " << config_filename << ", TRUE_EMISSION_SERIES" << std::endl;
            std::cerr << std::endl;
            exit(1);
        }
        if (g_eki.true_emissions[i] > 1.0e+20f) {
            std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                      << Color::RESET << "Unrealistically large emission at timestep " << (i+1)
                      << ": " << g_eki.true_emissions[i] << " Bq" << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
            std::cerr << "    Emission rate exceeds physically plausible values." << std::endl;
            std::cerr << "    Check units and magnitude." << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::GREEN << "Reference:" << Color::RESET << std::endl;
            std::cerr << "    Fukushima accident peak: ~1e+15 - 1e+17 Bq/hour" << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET
                      << " " << config_filename << ", TRUE_EMISSION_SERIES" << std::endl;
            std::cerr << std::endl;
            exit(1);
        }
    }

    // ===== VALIDATION: TIME_INTERVAL =====
    if (g_eki.time_interval <= 0.0f) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Invalid EKI_TIME_INTERVAL: " << g_eki.time_interval << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Time interval must be positive." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required value:" << Color::RESET << std::endl;
        std::cerr << "    EKI_TIME_INTERVAL > 0.0" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Typical values:" << Color::RESET << std::endl;
        std::cerr << "    - Fine resolution:   5-10 minutes" << std::endl;
        std::cerr << "    - Standard:          15-30 minutes" << std::endl;
        std::cerr << "    - Coarse:            1-3 hours" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " " << config_filename << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    std::cout << Color::GREEN << "done" << Color::RESET << std::endl;

    // Print essential EKI settings (condensed)
    std::cout << Color::BOLD << "EKI Configuration" << Color::RESET << std::endl;
    std::cout << "  Receptors          : " << Color::BOLD << g_eki.num_receptors << Color::RESET
              << " (radius: " << g_eki.receptor_capture_radius << "°)" << std::endl;
    std::cout << "  Emission timesteps : " << Color::BOLD << g_eki.true_emissions.size() << Color::RESET
              << " (" << g_eki.time_interval << " " << g_eki.time_unit << ")" << std::endl;
    std::cout << "  Ensemble size      : " << Color::BOLD << g_eki.ensemble_size << Color::RESET << std::endl;

    if (g_eki.memory_doctor_mode) {
        std::cout << "  Memory Doctor      : " << Color::YELLOW << "ON" << Color::RESET << std::endl;
    }
}

// ===========================================================================
// GRID RECEPTOR DEBUG MODE FUNCTIONS
// ===========================================================================

/******************************************************************************
 * @brief Initialize uniform grid of receptors for debugging/validation
 *
 * @details Creates a (2N+1)×(2N+1) square grid of receptors centered at the
 *          emission source location. Used in receptor-debug mode for detailed
 *          spatial analysis of particle dispersion patterns and validation
 *          against analytical solutions.
 *
 *          Grid structure:
 *          - Center: Source location (37°N, 141°E by default)
 *          - Extent: ±N grid points in lat/lon directions
 *          - Spacing: Uniform grid spacing in degrees
 *          - Example: grid_count=5, spacing=0.1° → 11×11=121 receptors
 *
 * @param[in] grid_count_param Grid extent (N receptors in each direction)
 *                             - Typical range: 5-10
 *                             - Total receptors = (2N+1)²
 * @param[in] grid_spacing_param Spacing between receptors (degrees)
 *                               - Typical range: 0.05-0.2°
 *                               - Approximate: 0.1° ≈ 11 km at mid-latitudes
 *
 * @pre CUDA device must be initialized
 * @pre Sufficient GPU memory for receptor arrays
 *
 * @post GPU arrays allocated and initialized:
 *       - d_grid_receptor_lats: Receptor latitudes (degrees N)
 *       - d_grid_receptor_lons: Receptor longitudes (degrees E)
 *       - d_grid_receptor_dose: Dose accumulation (initialized to 0)
 *       - d_grid_receptor_particle_count: Particle counts (initialized to 0)
 * @post Host storage vectors resized:
 *       - grid_receptor_observations
 *       - grid_receptor_particle_counts
 * @post Member variables set:
 *       - grid_count, grid_spacing, grid_receptor_total
 *
 * @algorithm
 *   1. Calculate total receptors = (2*grid_count + 1)²
 *   2. Generate receptor grid centered at source:
 *      for i in [-N, N]:
 *        for j in [-N, N]:
 *          lat = source_lat + i * grid_spacing
 *          lon = source_lon + j * grid_spacing
 *   3. Allocate GPU memory for receptor arrays
 *   4. Copy locations to GPU (cudaMemcpy)
 *   5. Initialize dose/count arrays to zero (cudaMemset)
 *   6. Resize host storage vectors
 *
 * @note Grid is always square and centered at source location
 * @note Large grids (>20×20) may impact performance due to memory overhead
 * @note Used exclusively in receptor-debug mode (not EKI mode)
 *
 * @memory GPU: 4 * total_receptors * sizeof(float) + 1 * total_receptors * sizeof(int)
 *         Example: 121 receptors = 2.4 KB total
 *
 * @see main_receptor_debug.cu for usage
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::initializeGridReceptors(int grid_count_param, float grid_spacing_param) {
    // Store grid parameters
    grid_count = grid_count_param;
    grid_spacing = grid_spacing_param;
    grid_receptor_total = (2 * grid_count + 1) * (2 * grid_count + 1);

    std::cout << Color::CYAN << "[SYSTEM] " << Color::RESET
              << "Initializing " << Color::BOLD << grid_receptor_total << Color::RESET
              << " grid receptors (" << (2*grid_count+1) << "×" << (2*grid_count+1)
              << ", spacing=" << grid_spacing << "°)" << std::endl;

    // Default source location (Fukushima coordinates)
    float source_lat = 37.0f;
    float source_lon = 141.0f;

    // Prepare host arrays for receptor locations
    std::vector<float> host_lats(grid_receptor_total);
    std::vector<float> host_lons(grid_receptor_total);

    // Generate uniform grid centered at source
    int receptor_idx = 0;
    for (int i = -grid_count; i <= grid_count; i++) {
        for (int j = -grid_count; j <= grid_count; j++) {
            float lat = source_lat + i * grid_spacing;
            float lon = source_lon + j * grid_spacing;

            host_lats[receptor_idx] = lat;
            host_lons[receptor_idx] = lon;
            receptor_idx++;
        }
    }

    // Allocate GPU memory for receptor data
    cudaMalloc(&d_grid_receptor_lats, grid_receptor_total * sizeof(float));
    cudaMalloc(&d_grid_receptor_lons, grid_receptor_total * sizeof(float));
    cudaMalloc(&d_grid_receptor_dose, grid_receptor_total * sizeof(float));
    cudaMalloc(&d_grid_receptor_particle_count, grid_receptor_total * sizeof(int));

    // Copy receptor locations to GPU
    cudaMemcpy(d_grid_receptor_lats, host_lats.data(), grid_receptor_total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid_receptor_lons, host_lons.data(), grid_receptor_total * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize GPU dose and particle count arrays to zero
    cudaMemset(d_grid_receptor_dose, 0, grid_receptor_total * sizeof(float));
    cudaMemset(d_grid_receptor_particle_count, 0, grid_receptor_total * sizeof(int));

    // Initialize host storage for observations
    grid_receptor_observations.resize(grid_receptor_total);
    grid_receptor_particle_counts.resize(grid_receptor_total);

    std::cout << Color::GREEN << "  " << Color::RESET
              << "Grid receptors initialized" << std::endl;
}

// ===========================================================================
// MODERNIZED CONFIG LOADING FUNCTIONS (Phase 1: 2025-10-17)
// ===========================================================================
// These functions implement the new modular configuration file structure
// described in docs/INPUT_MODERNIZATION_PLAN.md. Provides improved usability
// with self-documenting config files, logical grouping, and backward compatibility.

/******************************************************************************
 * @brief Load simulation parameters from modernized simulation.conf file
 *
 * @details Parses input/simulation.conf to load core simulation settings:
 *          - Temporal: time_end, time_step, vtk_output_frequency
 *          - Particles: total_particles
 *          - Atmosphere: rural_conditions, use_pasquill_gifford
 *          - Meteorology: use_gfs_data
 *          - Terminal: fixed_scroll_output
 *
 *          Part of Phase 1 input file modernization (2025-10-17).
 *
 * @pre input/simulation.conf must exist
 * @post Member variables populated: time_end, dt, freq_output, nop, isRural, isPG, isGFS
 * @post g_sim.fixedScrollOutput set
 * @post Configuration summary printed to console
 *
 * @see docs/INPUT_MODERNIZATION_PLAN.md for config file format
 * @see input/simulation.conf for configuration template
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::loadSimulationConfig() {
    std::cout << Color::CYAN << "[CONFIG] " << Color::RESET
              << "Loading simulation.conf... " << std::flush;

    // Load configuration file
    if (!g_config.loadConfig("input/simulation.conf")) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[FATAL ERROR] "
                  << Color::RESET << "Failed to load input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Possible causes:" << Color::RESET << std::endl;
        std::cerr << "    - File does not exist in the input/ directory" << std::endl;
        std::cerr << "    - Insufficient read permissions" << std::endl;
        std::cerr << "    - File is corrupted or locked by another process" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Solution:" << Color::RESET << std::endl;
        std::cerr << "    - Verify that 'input/simulation.conf' exists" << std::endl;
        std::cerr << "    - Check file permissions: chmod 644 input/simulation.conf" << std::endl;
        std::cerr << "    - Ensure you are running from the project root directory" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // ========== TEMPORAL SETTINGS ==========
    // Parse time_end (NO DEFAULT - explicit value required)
    std::string time_end_str = g_config.getString("time_end", "");
    if (time_end_str.empty()) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Missing required parameter: time_end" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    The time_end parameter is required to define simulation duration." << std::endl;
        std::cerr << "    This sets how long particles are tracked in the atmosphere." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required format:" << Color::RESET << std::endl;
        std::cerr << "    time_end: <positive number>  # in seconds" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Recommended values:" << Color::RESET << std::endl;
        std::cerr << "    - Short test:      3600  (1 hour)" << std::endl;
        std::cerr << "    - Standard:       21600  (6 hours)" << std::endl;
        std::cerr << "    - Long-range:     86400  (24 hours)" << std::endl;
        std::cerr << "    - Multi-day:     259200  (3 days)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET
                  << " input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }
    try {
        time_end = std::stof(time_end_str);
    } catch (const std::exception& e) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Cannot parse time_end value: '" << time_end_str << "'" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Value must be a valid floating-point number." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Valid examples:" << Color::RESET << std::endl;
        std::cerr << "    time_end: 21600" << std::endl;
        std::cerr << "    time_end: 21600.0" << std::endl;
        std::cerr << "    time_end: 2.16e4  # Scientific notation" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET
                  << " input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // Parse time_step (NO DEFAULT - explicit value required)
    std::string dt_str = g_config.getString("time_step", "");
    if (dt_str.empty()) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Missing required parameter: time_step" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    The time_step parameter is required for particle advancement." << std::endl;
        std::cerr << "    This controls temporal accuracy vs. computational cost." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required format:" << Color::RESET << std::endl;
        std::cerr << "    time_step: <positive number>  # in seconds" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Recommended values:" << Color::RESET << std::endl;
        std::cerr << "    - High accuracy:    10-50  seconds (slower)" << std::endl;
        std::cerr << "    - Balanced:        50-100  seconds (recommended)" << std::endl;
        std::cerr << "    - Fast:           100-300  seconds (less accurate)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET
                  << " input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }
    try {
        dt = std::stof(dt_str);
    } catch (const std::exception& e) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Cannot parse time_step value: '" << dt_str << "'" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Value must be a valid floating-point number." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Valid examples:" << Color::RESET << std::endl;
        std::cerr << "    time_step: 100" << std::endl;
        std::cerr << "    time_step: 100.0" << std::endl;
        std::cerr << "    time_step: 1.0e2  # Scientific notation" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET
                  << " input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // Parse vtk_output_frequency (NO DEFAULT - explicit value required)
    std::string freq_str = g_config.getString("vtk_output_frequency", "");
    if (freq_str.empty()) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Missing required parameter: vtk_output_frequency" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    The vtk_output_frequency parameter controls visualization output." << std::endl;
        std::cerr << "    This determines how often VTK files are written." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required format:" << Color::RESET << std::endl;
        std::cerr << "    vtk_output_frequency: <positive integer>  # timesteps per output" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Recommended values:" << Color::RESET << std::endl;
        std::cerr << "    - Every timestep:    1  (detailed, large files)" << std::endl;
        std::cerr << "    - Every 10 steps:   10  (balanced)" << std::endl;
        std::cerr << "    - Sparse output:   100  (minimal disk usage)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET
                  << " input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }
    try {
        freq_output = std::stoi(freq_str);
    } catch (const std::exception& e) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Cannot parse vtk_output_frequency value: '" << freq_str << "'" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Value must be a valid integer." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Valid examples:" << Color::RESET << std::endl;
        std::cerr << "    vtk_output_frequency: 1" << std::endl;
        std::cerr << "    vtk_output_frequency: 10" << std::endl;
        std::cerr << "    vtk_output_frequency: 100" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET
                  << " input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // ===== VALIDATION: time_end =====
    if (time_end <= 0.0f) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Invalid time_end: " << time_end << " seconds" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Simulation duration must be positive." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required value:" << Color::RESET << std::endl;
        std::cerr << "    time_end > 0 (in seconds)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Recommended range:" << Color::RESET << std::endl;
        std::cerr << "    - Short test: 3600 (1 hour)" << std::endl;
        std::cerr << "    - Medium: 21600 (6 hours)" << std::endl;
        std::cerr << "    - Long: 86400 (24 hours)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }
    if (time_end > 604800.0f) {  // 7 days
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Excessively long simulation time: " << time_end << " seconds ("
                  << (time_end / 86400.0f) << " days)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Simulations longer than 7 days may be impractical due to:" << std::endl;
        std::cerr << "    - Excessive computation time" << std::endl;
        std::cerr << "    - Meteorological data availability" << std::endl;
        std::cerr << "    - Accumulated numerical errors" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required value:" << Color::RESET << std::endl;
        std::cerr << "    time_end <= 604800 seconds (7 days)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // ===== VALIDATION: time_step (dt) =====
    if (dt <= 0.0f) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Invalid time_step: " << dt << " seconds" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Time step must be positive." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required value:" << Color::RESET << std::endl;
        std::cerr << "    time_step > 0 (in seconds)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Recommended range:" << Color::RESET << std::endl;
        std::cerr << "    - High accuracy: 10-50 seconds" << std::endl;
        std::cerr << "    - Balanced: 50-100 seconds (recommended)" << std::endl;
        std::cerr << "    - Fast: 100-300 seconds (less accurate)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }
    if (dt > 3600.0f) {  // 1 hour
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Time step too large: " << dt << " seconds" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Large time steps cause numerical instability and poor accuracy." << std::endl;
        std::cerr << "    Particles may skip over important meteorological features." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required value:" << Color::RESET << std::endl;
        std::cerr << "    time_step <= 3600 seconds (1 hour)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Recommended:" << Color::RESET << " 100 seconds for good balance" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }
    if (dt >= time_end) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Time step must be smaller than simulation duration" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Current values:" << Color::RESET << std::endl;
        std::cerr << "    time_step = " << dt << " seconds" << std::endl;
        std::cerr << "    time_end  = " << time_end << " seconds" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required:" << Color::RESET << std::endl;
        std::cerr << "    time_step < time_end" << std::endl;
        std::cerr << "    Suggested: time_step <= time_end / 10" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Example fix:" << Color::RESET << std::endl;
        std::cerr << "    If time_end = " << time_end << " seconds" << std::endl;
        std::cerr << "    Then time_step = " << (time_end / 100.0f) << " seconds (or smaller)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // ===== VALIDATION: vtk_output_frequency =====
    if (freq_output <= 0) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Invalid vtk_output_frequency: " << freq_output << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Output frequency must be positive." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required value:" << Color::RESET << std::endl;
        std::cerr << "    vtk_output_frequency >= 1" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Recommended values:" << Color::RESET << std::endl;
        std::cerr << "    - Every timestep: 1 (most detailed, large files)" << std::endl;
        std::cerr << "    - Every 10th step: 10 (balanced)" << std::endl;
        std::cerr << "    - Every 100th step: 100 (minimal output)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }
    if (freq_output > 1000) {
        std::cerr << std::endl << Color::YELLOW << Color::BOLD << "[WARNING] "
                  << Color::RESET << "Very sparse output frequency: " << freq_output << std::endl;
        std::cerr << "  This may result in insufficient visualization data." << std::endl;
        std::cerr << "  Consider using a smaller value (e.g., 10-100) for better analysis." << std::endl;
        std::cerr << std::endl;
    }

    // ========== PARTICLE SETTINGS ==========
    // Parse total_particles (NO DEFAULT - explicit value required)
    std::string nop_str = g_config.getString("total_particles", "");
    if (nop_str.empty()) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Missing required parameter: total_particles" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    The total_particles parameter is required for simulation." << std::endl;
        std::cerr << "    This determines statistical quality of results." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required format:" << Color::RESET << std::endl;
        std::cerr << "    total_particles: <positive integer>  # number of particles" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Recommended values:" << Color::RESET << std::endl;
        std::cerr << "    - Quick test:       1,000  particles" << std::endl;
        std::cerr << "    - Standard:        10,000  particles (good balance)" << std::endl;
        std::cerr << "    - High quality:   100,000  particles" << std::endl;
        std::cerr << "    - Production:   1,000,000  particles (requires GPU)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET
                  << " input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }
    try {
        nop = std::stoi(nop_str);
    } catch (const std::exception& e) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Cannot parse total_particles value: '" << nop_str << "'" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Value must be a valid integer." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Valid examples:" << Color::RESET << std::endl;
        std::cerr << "    total_particles: 10000" << std::endl;
        std::cerr << "    total_particles: 100000" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET
                  << " input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // ===== VALIDATION: total_particles =====
    if (nop <= 0) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Invalid total_particles: " << nop << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Particle count must be positive." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required value:" << Color::RESET << std::endl;
        std::cerr << "    total_particles >= 100 (minimum for meaningful statistics)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Recommended range:" << Color::RESET << std::endl;
        std::cerr << "    - Quick test: 1,000 particles" << std::endl;
        std::cerr << "    - Standard: 10,000 particles (good balance)" << std::endl;
        std::cerr << "    - High quality: 100,000 particles" << std::endl;
        std::cerr << "    - Production: 1,000,000 particles (requires GPU)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }
    if (nop < 100) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Too few particles: " << nop << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Fewer than 100 particles produces unreliable statistics." << std::endl;
        std::cerr << "    Results will be dominated by random sampling noise." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required value:" << Color::RESET << std::endl;
        std::cerr << "    total_particles >= 100" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Recommended:" << Color::RESET << " At least 1,000 particles" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }
    if (nop > 100000000) {  // 100 million
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Excessive particle count: " << nop << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    This many particles will cause:" << std::endl;
        std::cerr << "    - GPU memory exhaustion" << std::endl;
        std::cerr << "    - Extremely long computation times" << std::endl;
        std::cerr << "    - Potential numerical overflow" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required value:" << Color::RESET << std::endl;
        std::cerr << "    total_particles <= 100,000,000" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Typical maximum:" << Color::RESET << " 10,000,000 particles" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // Note: Atmospheric conditions and meteorological data are hardcoded in v1.0
    // isRural=1, isPG=1, isGFS=1 (set in loadLegacyConfig)

    // ========== TERMINAL OUTPUT ==========
    // Parse fixed_scroll_output (NO DEFAULT - explicit value required)
    std::string fixedScroll_str = g_config.getString("fixed_scroll_output", "");
    if (fixedScroll_str.empty()) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Missing required parameter: fixed_scroll_output" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    The fixed_scroll_output parameter controls terminal output style." << std::endl;
        std::cerr << "    This affects how simulation progress is displayed." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required format:" << std::endl;
        std::cerr << "    fixed_scroll_output: <0 or 1>" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Valid values:" << Color::RESET << std::endl;
        std::cerr << "    0 = Continuous scroll (full history visible)" << std::endl;
        std::cerr << "    1 = Fixed-height (cleaner, stays within terminal)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Recommended:" << Color::RESET << std::endl;
        std::cerr << "    fixed_scroll_output: 1  # Cleaner output" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET
                  << " input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }
    try {
        g_sim.fixedScrollOutput = std::stoi(fixedScroll_str);
    } catch (const std::exception& e) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Cannot parse fixed_scroll_output value: '" << fixedScroll_str << "'" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Value must be either 0 or 1." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET
                  << " input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // ===== VALIDATION: fixed_scroll_output =====
    if (g_sim.fixedScrollOutput != 0 && g_sim.fixedScrollOutput != 1) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Invalid fixed_scroll_output: " << g_sim.fixedScrollOutput << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    This is a boolean flag - must be 0 or 1." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Valid values:" << Color::RESET << std::endl;
        std::cerr << "    0 = Continuous scroll (full history visible)" << std::endl;
        std::cerr << "    1 = Fixed-height (cleaner, stays within terminal)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // ========== VTK VISUALIZATION OUTPUT ==========
    // Parse enable_single_mode_vtk (NO DEFAULT - explicit value required)
    std::string vtk_single_str = g_config.getString("enable_single_mode_vtk", "");
    if (vtk_single_str.empty()) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Missing required parameter: enable_single_mode_vtk" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    The enable_single_mode_vtk parameter controls VTK output for initial simulation." << std::endl;
        std::cerr << "    This affects whether visualization files are generated for the 'truth' run." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required format:" << std::endl;
        std::cerr << "    enable_single_mode_vtk: <0 or 1>" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Valid values:" << Color::RESET << std::endl;
        std::cerr << "    0 = Disable VTK output (faster, no visualization)" << std::endl;
        std::cerr << "    1 = Enable VTK output (useful for visualization)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Recommended:" << Color::RESET << std::endl;
        std::cerr << "    enable_single_mode_vtk: 1  # Visualize initial simulation" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET
                  << " input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }
    try {
        config_enable_single_mode_vtk = (std::stoi(vtk_single_str) != 0);
    } catch (const std::exception& e) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Cannot parse enable_single_mode_vtk value: '" << vtk_single_str << "'" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Value must be either 0 or 1." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET
                  << " input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // ===== VALIDATION: enable_single_mode_vtk =====
    int vtk_single_int = std::stoi(vtk_single_str);
    if (vtk_single_int != 0 && vtk_single_int != 1) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Invalid enable_single_mode_vtk: " << vtk_single_int << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    This is a boolean flag - must be 0 or 1." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Valid values:" << Color::RESET << std::endl;
        std::cerr << "    0 = Disable VTK output (faster, no visualization)" << std::endl;
        std::cerr << "    1 = Enable VTK output (useful for visualization)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // Parse enable_ensemble_mode_vtk (NO DEFAULT - explicit value required)
    std::string vtk_ensemble_str = g_config.getString("enable_ensemble_mode_vtk", "");
    if (vtk_ensemble_str.empty()) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Missing required parameter: enable_ensemble_mode_vtk" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    The enable_ensemble_mode_vtk parameter controls VTK output for ensemble iterations." << std::endl;
        std::cerr << "    This affects whether visualization files are generated during EKI iterations." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required format:" << std::endl;
        std::cerr << "    enable_ensemble_mode_vtk: <0 or 1>" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Valid values:" << Color::RESET << std::endl;
        std::cerr << "    0 = Disable VTK output (recommended for performance)" << std::endl;
        std::cerr << "    1 = Enable VTK output (large files, slower)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Recommended:" << Color::RESET << std::endl;
        std::cerr << "    enable_ensemble_mode_vtk: 0  # Disable for performance" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Note:" << Color::RESET << std::endl;
        std::cerr << "    - Ensemble VTK files can be very large (100s of GB)" << std::endl;
        std::cerr << "    - Only final iteration is saved if enabled" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET
                  << " input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }
    try {
        config_enable_ensemble_mode_vtk = (std::stoi(vtk_ensemble_str) != 0);
    } catch (const std::exception& e) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Cannot parse enable_ensemble_mode_vtk value: '" << vtk_ensemble_str << "'" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Value must be either 0 or 1." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET
                  << " input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // ===== VALIDATION: enable_ensemble_mode_vtk =====
    int vtk_ensemble_int = std::stoi(vtk_ensemble_str);
    if (vtk_ensemble_int != 0 && vtk_ensemble_int != 1) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Invalid enable_ensemble_mode_vtk: " << vtk_ensemble_int << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    This is a boolean flag - must be 0 or 1." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Valid values:" << Color::RESET << std::endl;
        std::cerr << "    0 = Disable VTK output (recommended for performance)" << std::endl;
        std::cerr << "    1 = Enable VTK output (large files, slower)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " input/simulation.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    std::cout << Color::GREEN << "done" << Color::RESET << std::endl;

    // ========== PRINT CONFIGURATION SUMMARY ==========
    std::cout << Color::BOLD << "Simulation Configuration" << Color::RESET << std::endl;

    // Temporal settings
    std::cout << "  Time settings      : " << Color::BOLD << time_end << "s" << Color::RESET
              << " (dt=" << dt << "s, "
              << "output_freq=" << freq_output << ")" << std::endl;

    // Particle count
    std::cout << "  Particles          : " << Color::BOLD << nop << Color::RESET << std::endl;

    // Atmospheric conditions
    std::cout << "  Atmosphere         : "
              << (isRural ? "Rural" : "Urban") << ", "
              << (isPG ? "Pasquill-Gifford" : "Briggs-McElroy-Pooler") << std::endl;

    // Meteorological data
    std::cout << "  Meteorology        : " << (isGFS ? "GFS" : "LDAPS") << std::endl;

    // Terminal output
    std::cout << "  Terminal output    : "
              << (g_sim.fixedScrollOutput ? "Fixed-scroll" : "Continuous-scroll") << std::endl;

    // VTK visualization output
    std::cout << "  VTK output         : "
              << "Single=" << (config_enable_single_mode_vtk ? (std::string(Color::GREEN) + "ON") : (std::string(Color::YELLOW) + "OFF")) << Color::RESET
              << ", Ensemble=" << (config_enable_ensemble_mode_vtk ? (std::string(Color::GREEN) + "ON") : (std::string(Color::YELLOW) + "OFF")) << Color::RESET
              << std::endl;
}

/******************************************************************************
 * @brief Load physics model configuration from physics.conf
 *
 * @details Parses input/physics.conf to configure physics model switches:
 *          - dry_deposition_model: Dry deposition (On/Off)
 *          - wet_deposition_model: Wet deposition (On/Off)
 *          - radioactive_decay_model: Decay (On/Off)
 *
 *          Note: turbulence_model is hardcoded to 0 (not implemented in v1.0)
 *
 * @pre input/physics.conf must exist
 * @post Physics switches set: g_drydep, g_wetdep, g_raddecay
 * @post g_turb_switch hardcoded to 0
 * @post Configuration summary printed to console
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::loadPhysicsConfig() {
    std::cout << Color::CYAN << "[SYSTEM] " << Color::RESET
              << "Loading physics configuration... " << std::flush;

    // Load physics.conf using ConfigReader
    ConfigReader physics_config;
    if (!physics_config.loadConfig("input/physics.conf")) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[FATAL ERROR] "
                  << Color::RESET << "Failed to load input/physics.conf" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Possible causes:" << Color::RESET << std::endl;
        std::cerr << "    - File does not exist in the input/ directory" << std::endl;
        std::cerr << "    - Insufficient read permissions" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Solution:" << Color::RESET << std::endl;
        std::cerr << "    - Verify that 'input/physics.conf' exists" << std::endl;
        std::cerr << "    - Check file permissions: chmod 644 input/physics.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // Parse physics model switches (NO DEFAULTS - all must be explicitly provided)
    // Note: turbulence_model is hardcoded to 0 (not implemented in v1.0)

    // Helper lambda to parse On/Off values
    auto parseOnOff = [](const std::string& value, const std::string& param_name) -> int {
        if (value == "On" || value == "on" || value == "ON") {
            return 1;
        } else if (value == "Off" || value == "off" || value == "OFF") {
            return 0;
        } else {
            std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                      << Color::RESET << "Invalid value for " << param_name << ": '" << value << "'" << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
            std::cerr << "    Value must be 'On' or 'Off' (case-insensitive)." << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::CYAN << "Valid values:" << Color::RESET << std::endl;
            std::cerr << "    On  = Enabled" << std::endl;
            std::cerr << "    Off = Disabled" << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " input/physics.conf" << std::endl;
            std::cerr << std::endl;
            exit(1);
        }
        return 0;
    };

    // Parse dry_deposition_model (NO DEFAULT)
    std::string drydep_str = physics_config.getString("dry_deposition_model", "");
    if (drydep_str.empty()) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Missing required parameter: dry_deposition_model" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    The dry_deposition_model parameter controls gravitational settling." << std::endl;
        std::cerr << "    This is important for particulate matter transport." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required format:" << Color::RESET << std::endl;
        std::cerr << "    dry_deposition_model: <On or Off>" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Valid values:" << Color::RESET << std::endl;
        std::cerr << "    Off = Disabled (particles do not settle)" << std::endl;
        std::cerr << "    On  = Enabled (gravitational settling and surface deposition)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Recommended:" << Color::RESET << std::endl;
        std::cerr << "    dry_deposition_model: On  # For particulate matter" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " input/physics.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }
    g_drydep = parseOnOff(drydep_str, "dry_deposition_model");

    // Parse wet_deposition_model (NO DEFAULT)
    std::string wetdep_str = physics_config.getString("wet_deposition_model", "");
    if (wetdep_str.empty()) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Missing required parameter: wet_deposition_model" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    The wet_deposition_model parameter controls precipitation removal." << std::endl;
        std::cerr << "    This is critical during rain or snow events." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required format:" << Color::RESET << std::endl;
        std::cerr << "    wet_deposition_model: <On or Off>" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Valid values:" << Color::RESET << std::endl;
        std::cerr << "    Off = Disabled (no precipitation removal)" << std::endl;
        std::cerr << "    On  = Enabled (removal by rain and snow)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Recommended:" << Color::RESET << std::endl;
        std::cerr << "    wet_deposition_model: On  # If precipitation expected" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " input/physics.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }
    g_wetdep = parseOnOff(wetdep_str, "wet_deposition_model");

    // Parse radioactive_decay_model (NO DEFAULT)
    std::string raddecay_str = physics_config.getString("radioactive_decay_model", "");
    if (raddecay_str.empty()) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "Missing required parameter: radioactive_decay_model" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    The radioactive_decay_model parameter controls CRAM decay computation." << std::endl;
        std::cerr << "    This is essential for radionuclide transport simulations." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required format:" << Color::RESET << std::endl;
        std::cerr << "    radioactive_decay_model: <On or Off>" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Valid values:" << Color::RESET << std::endl;
        std::cerr << "    Off = Disabled (no radioactive decay)" << std::endl;
        std::cerr << "    On  = Enabled (CRAM decay computation)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Recommended:" << Color::RESET << std::endl;
        std::cerr << "    radioactive_decay_model: On  # Keep ON for radionuclides" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " input/physics.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }
    g_raddecay = parseOnOff(raddecay_str, "radioactive_decay_model");

    // Note: Validation is handled by parseOnOff() lambda above
    // No additional validation needed - values are guaranteed to be 0 or 1

    std::cout << Color::GREEN << "done" << Color::RESET << std::endl;

    // Print physics model status summary
    std::cout << Color::BOLD << "Physics Models" << Color::RESET << std::endl;
    std::cout << "  Turbulence         : " << (g_turb_switch ? Color::GREEN : Color::YELLOW)
              << (g_turb_switch ? "ON" : "OFF") << Color::RESET << std::endl;
    std::cout << "  Dry Deposition     : " << (g_drydep ? Color::GREEN : Color::YELLOW)
              << (g_drydep ? "ON" : "OFF") << Color::RESET << std::endl;
    std::cout << "  Wet Deposition     : " << (g_wetdep ? Color::GREEN : Color::YELLOW)
              << (g_wetdep ? "ON" : "OFF") << Color::RESET << std::endl;
    std::cout << "  Radioactive Decay  : " << (g_raddecay ? Color::GREEN : Color::YELLOW)
              << (g_raddecay ? "ON" : "OFF") << Color::RESET << std::endl;
}

/******************************************************************************
 * @brief Load source locations from source.conf file
 *
 * @details Parses input/source.conf to load emission source coordinates.
 *          Format: LONGITUDE LATITUDE HEIGHT (space-separated, degrees/meters)
 *
 *          Example:
 *            # Source 1: Fukushima Daiichi
 *            141.0 37.0 20.0
 *
 * @pre input/source.conf must exist
 * @post sources vector populated with Source structs (lon, lat, height)
 * @post At least one source must be defined (validation check)
 *
 * @note Lines starting with # are comments
 * @note Empty lines ignored
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::loadSourceConfig() {
    std::cout << Color::CYAN << "[SYSTEM] " << Color::RESET
              << "Loading source locations... " << std::flush;

    // Construct file path (NO DEFAULT for input_base_path)
    std::string input_base = g_config.getString("input_base_path", "");
    if (input_base.empty()) {
        // Default behavior: use "./input/" if not specified
        // This is acceptable as it's a path convention, not a physical parameter
        input_base = "./input/";
    }
    std::string source_file_path = input_base + "source.conf";

    FILE* sourceFile = fopen(source_file_path.c_str(), "r");

    if (!sourceFile) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[FATAL ERROR] "
                  << Color::RESET << "Failed to open " << source_file_path << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Possible causes:" << Color::RESET << std::endl;
        std::cerr << "    - File does not exist in the input/ directory" << std::endl;
        std::cerr << "    - Insufficient read permissions" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Solution:" << Color::RESET << std::endl;
        std::cerr << "    - Verify that 'input/source.conf' exists" << std::endl;
        std::cerr << "    - Check file permissions: chmod 644 input/source.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    char buffer[256];
    int line_number = 0;

    // Clear existing sources
    sources.clear();

    while (fgets(buffer, sizeof(buffer), sourceFile)) {
        line_number++;

        // Skip comment lines starting with #
        if (buffer[0] == '#') continue;

        // Skip empty lines
        bool is_empty = true;
        for (int i = 0; buffer[i] != '\0'; i++) {
            if (buffer[i] != ' ' && buffer[i] != '\t' &&
                buffer[i] != '\n' && buffer[i] != '\r') {
                is_empty = false;
                break;
            }
        }
        if (is_empty) continue;

        // Stop parsing when encountering a section header (e.g., [GRID_CONFIG])
        // Source locations are defined before any section headers
        if (buffer[0] == '[') {
            break;
        }

        // Parse source location: LON LAT HEIGHT
        Source src;
        int parsed = sscanf(buffer, "%f %f %f", &src.lon, &src.lat, &src.height);

        if (parsed != 3) {
            fclose(sourceFile);
            std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                      << Color::RESET << "Invalid format at line " << line_number
                      << " in source.conf" << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::YELLOW << "Invalid line:" << Color::RESET << std::endl;
            std::cerr << "    " << buffer;
            std::cerr << std::endl;
            std::cerr << "  " << Color::CYAN << "Required format:" << Color::RESET << std::endl;
            std::cerr << "    LONGITUDE LATITUDE HEIGHT" << std::endl;
            std::cerr << "    (space-separated, degrees and meters)" << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::GREEN << "Example:" << Color::RESET << std::endl;
            std::cerr << "    129.48 35.71 100.0" << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " input/source.conf" << std::endl;
            std::cerr << std::endl;
            exit(1);
        }

        // ===== VALIDATION: Longitude =====
        if (src.lon < -180.0f || src.lon > 180.0f) {
            fclose(sourceFile);
            std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                      << Color::RESET << "Invalid longitude: " << src.lon << "° at line "
                      << line_number << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
            std::cerr << "    Longitude must be in valid geographic range." << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::CYAN << "Required range:" << Color::RESET << std::endl;
            std::cerr << "    -180.0 <= longitude <= 180.0 (degrees)" << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::GREEN << "Examples:" << Color::RESET << std::endl;
            std::cerr << "    - Tokyo:      139.69°E" << std::endl;
            std::cerr << "    - New York:   -74.01°E (or 285.99°W)" << std::endl;
            std::cerr << "    - Fukushima:  141.00°E" << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " input/source.conf, line "
                      << line_number << std::endl;
            std::cerr << std::endl;
            exit(1);
        }

        // ===== VALIDATION: Latitude =====
        if (src.lat < -90.0f || src.lat > 90.0f) {
            fclose(sourceFile);
            std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                      << Color::RESET << "Invalid latitude: " << src.lat << "° at line "
                      << line_number << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
            std::cerr << "    Latitude must be in valid geographic range." << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::CYAN << "Required range:" << Color::RESET << std::endl;
            std::cerr << "    -90.0 <= latitude <= 90.0 (degrees)" << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::GREEN << "Examples:" << Color::RESET << std::endl;
            std::cerr << "    - Equator:    0.00°N" << std::endl;
            std::cerr << "    - Tokyo:      35.69°N" << std::endl;
            std::cerr << "    - Fukushima:  37.00°N" << std::endl;
            std::cerr << "    - South Pole: -90.00°N" << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " input/source.conf, line "
                      << line_number << std::endl;
            std::cerr << std::endl;
            exit(1);
        }

        // ===== VALIDATION: Height =====
        if (src.height < 0.0f) {
            fclose(sourceFile);
            std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                      << Color::RESET << "Invalid height: " << src.height << " m at line "
                      << line_number << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
            std::cerr << "    Release height cannot be negative (below ground level)." << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::CYAN << "Required range:" << Color::RESET << std::endl;
            std::cerr << "    height >= 0.0 (meters above ground level)" << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::GREEN << "Typical values:" << Color::RESET << std::endl;
            std::cerr << "    - Ground release:    0-10 m" << std::endl;
            std::cerr << "    - Building release:  20-100 m" << std::endl;
            std::cerr << "    - Stack release:     100-500 m" << std::endl;
            std::cerr << "    - Elevated source:   500-3000 m" << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " input/source.conf, line "
                      << line_number << std::endl;
            std::cerr << std::endl;
            exit(1);
        }
        if (src.height > 20000.0f) {
            fclose(sourceFile);
            std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                      << Color::RESET << "Excessive height: " << src.height << " m at line "
                      << line_number << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
            std::cerr << "    Release height exceeds practical atmospheric boundary layer." << std::endl;
            std::cerr << "    Heights above 20 km are typically not relevant for dispersion modeling." << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::CYAN << "Required range:" << Color::RESET << std::endl;
            std::cerr << "    height <= 20000.0 meters (20 km)" << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::GREEN << "Reference:" << Color::RESET << std::endl;
            std::cerr << "    - Troposphere top: ~12 km" << std::endl;
            std::cerr << "    - Stratosphere begins: ~12-15 km" << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " input/source.conf, line "
                      << line_number << std::endl;
            std::cerr << std::endl;
            exit(1);
        }

        sources.push_back(src);
    }

    fclose(sourceFile);

    // Validation: at least one source must be defined
    if (sources.empty()) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "No valid sources found in source.conf" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    At least one emission source must be defined." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Solution:" << Color::RESET << std::endl;
        std::cerr << "    Add at least one source line in the format:" << std::endl;
        std::cerr << "    LONGITUDE LATITUDE HEIGHT" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Example:" << Color::RESET << std::endl;
        std::cerr << "    # Fukushima Daiichi Nuclear Power Plant" << std::endl;
        std::cerr << "    141.0 37.0 20.0" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " input/source.conf" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    std::cout << Color::GREEN << "done" << Color::RESET << std::endl;

    // Print loaded sources summary
    std::cout << Color::BOLD << "Source Locations" << Color::RESET << std::endl;
    for (size_t i = 0; i < sources.size(); i++) {
        std::cout << "  Source " << (i+1) << "            : "
                  << sources[i].lon << "°E, "
                  << sources[i].lat << "°N, "
                  << sources[i].height << "m" << std::endl;
    }
}

/******************************************************************************
 * @brief Load nuclide configuration from nuclides.conf
 *
 * @details Parses nuclide properties with backward compatibility for legacy formats.
 *          Tries files in order:
 *          1. input/nuclides.conf (new format)
 *          2. input/nuclides_config_1.txt (legacy single nuclide)
 *          3. input/nuclides_config_60.txt (legacy 60-nuclide chain)
 *
 *          New format (space-separated):
 *            NUCLIDE_NAME DECAY_CONSTANT(s^-1) DEPOSITION_VELOCITY(m/s)
 *
 *          Legacy format (comma-separated):
 *            NUCLIDE_NAME,DECAY_CONSTANT,RATIO
 *
 * @pre At least one nuclide configuration file must exist
 * @post decayConstants vector populated with decay constants (s^-1)
 * @post drydepositionVelocity vector populated with deposition velocities (m/s)
 * @post g_num_nuclides set to number of nuclides loaded
 *
 * @note Decay constants forced to positive values (fabs applied)
 * @note Legacy format uses default deposition velocity = 1.0 m/s
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::loadNuclidesConfig() {
    std::cout << Color::CYAN << "[SYSTEM] " << Color::RESET
              << "Loading nuclide configuration... " << std::flush;

    FILE* nuclideFile = nullptr;
    std::string filename;

    // Try new format first (nuclides.conf)
    filename = "input/nuclides.conf";
    nuclideFile = fopen(filename.c_str(), "r");

    // Fall back to legacy format (nuclides_config_1.txt)
    if (!nuclideFile) {
        filename = "input/nuclides_config_1.txt";
        nuclideFile = fopen(filename.c_str(), "r");
    }

    // Fall back to 60-nuclide chain if available
    if (!nuclideFile) {
        filename = "input/nuclides_config_60.txt";
        nuclideFile = fopen(filename.c_str(), "r");
    }

    if (!nuclideFile) {
        std::cerr << std::endl << Color::RED << "[ERROR] " << Color::RESET
                  << "Cannot open nuclide configuration file" << std::endl;
        std::cerr << "  Tried: input/nuclides.conf" << std::endl;
        std::cerr << "         input/nuclides_config_1.txt" << std::endl;
        std::cerr << "         input/nuclides_config_60.txt" << std::endl;
        exit(1);
    }

    // Clear existing data
    decayConstants.clear();
    drydepositionVelocity.clear();

    char buffer[256];
    int line_number = 0;
    int nuclide_count = 0;

    while (fgets(buffer, sizeof(buffer), nuclideFile)) {
        line_number++;

        // Skip comments and empty lines
        if (buffer[0] == '#' || buffer[0] == '\n' || buffer[0] == '\r') {
            continue;
        }

        // Remove trailing newline
        buffer[strcspn(buffer, "\n\r")] = '\0';

        // Skip empty lines (after trimming)
        if (strlen(buffer) == 0) {
            continue;
        }

        char nuclide_name[64];
        float decay_const, dep_vel;

        // Try new format first (space-separated)
        int parsed = sscanf(buffer, "%s %f %f", nuclide_name, &decay_const, &dep_vel);

        if (parsed == 3) {
            // Successfully parsed new format
            decayConstants.push_back(fabs(decay_const));  // Ensure positive
            drydepositionVelocity.push_back(dep_vel);
            nuclide_count++;
        }
        else {
            // Try legacy comma-separated format
            float legacy_ratio;
            parsed = sscanf(buffer, "%[^,],%f,%f", nuclide_name, &decay_const, &legacy_ratio);

            if (parsed == 3) {
                // Successfully parsed legacy format
                // LEGACY FORMAT WARNING: Using 1.0 m/s as deposition velocity
                // because legacy format doesn't specify it (3rd column is ratio, not dep_vel)
                std::cerr << Color::YELLOW << "[LEGACY FORMAT] " << Color::RESET
                          << "Line " << line_number << ": Using default deposition velocity 1.0 m/s" << std::endl;
                std::cerr << "  Consider migrating to new format: NUCLIDE DECAY_CONST DEP_VEL" << std::endl;
                decayConstants.push_back(fabs(decay_const));  // Ensure positive
                drydepositionVelocity.push_back(1.0f);  // Hardcoded for legacy compatibility
                nuclide_count++;
            }
            else {
                std::cerr << std::endl << Color::YELLOW << "[WARNING] " << Color::RESET
                          << "Failed to parse line " << line_number << " in " << filename << std::endl;
                std::cerr << "  Line: " << buffer << std::endl;
                continue;
            }
        }
    }

    fclose(nuclideFile);

    // Set global nuclide count
    g_num_nuclides = nuclide_count;

    if (nuclide_count == 0) {
        std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                  << Color::RESET << "No valid nuclides loaded from " << filename << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Nuclide configuration file exists but contains no valid entries." << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Required format:" << Color::RESET << std::endl;
        std::cerr << "    NUCLIDE_NAME DECAY_CONSTANT DEPOSITION_VELOCITY" << std::endl;
        std::cerr << "    (space-separated)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::GREEN << "Example:" << Color::RESET << std::endl;
        std::cerr << "    Cs137 7.30e-10 0.01" << std::endl;
        std::cerr << "    I131  9.97e-07 0.02" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " " << filename << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    // ===== VALIDATION: Check all decay constants =====
    for (size_t i = 0; i < decayConstants.size(); i++) {
        if (decayConstants[i] < 0.0f) {
            std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                      << Color::RESET << "Negative decay constant for nuclide " << (i+1)
                      << ": " << decayConstants[i] << " s⁻¹" << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
            std::cerr << "    Decay constants must be non-negative." << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::CYAN << "Physical meaning:" << Color::RESET << std::endl;
            std::cerr << "    Decay constant λ relates to half-life: t₁/₂ = ln(2)/λ" << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::GREEN << "Typical ranges:" << Color::RESET << std::endl;
            std::cerr << "    - Stable isotopes:  0.0 s⁻¹" << std::endl;
            std::cerr << "    - Long-lived (Cs-137):  7.3e-10 s⁻¹ (t₁/₂ = 30 years)" << std::endl;
            std::cerr << "    - Medium-lived (I-131): 9.97e-07 s⁻¹ (t₁/₂ = 8 days)" << std::endl;
            std::cerr << "    - Short-lived (Xe-133): 1.52e-06 s⁻¹ (t₁/₂ = 5 days)" << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " " << filename << std::endl;
            std::cerr << std::endl;
            exit(1);
        }
        if (decayConstants[i] > 1.0f) {
            std::cerr << std::endl << Color::YELLOW << Color::BOLD << "[WARNING] "
                      << Color::RESET << "Very large decay constant for nuclide " << (i+1)
                      << ": " << decayConstants[i] << " s⁻¹" << std::endl;
            std::cerr << "  This corresponds to a half-life of " << (0.693147f / decayConstants[i])
                      << " seconds." << std::endl;
            std::cerr << "  Such short-lived nuclides may decay before significant transport occurs." << std::endl;
            std::cerr << std::endl;
        }
    }

    // ===== VALIDATION: Check all deposition velocities =====
    for (size_t i = 0; i < drydepositionVelocity.size(); i++) {
        if (drydepositionVelocity[i] < 0.0f) {
            std::cerr << std::endl << Color::RED << Color::BOLD << "[INPUT ERROR] "
                      << Color::RESET << "Negative deposition velocity for nuclide " << (i+1)
                      << ": " << drydepositionVelocity[i] << " m/s" << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
            std::cerr << "    Deposition velocities must be non-negative." << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::CYAN << "Physical meaning:" << Color::RESET << std::endl;
            std::cerr << "    Rate at which particles settle to the ground surface." << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::GREEN << "Typical ranges:" << Color::RESET << std::endl;
            std::cerr << "    - Gases:           0.001-0.01 m/s" << std::endl;
            std::cerr << "    - Small particles: 0.001-0.01 m/s" << std::endl;
            std::cerr << "    - Large particles: 0.01-0.1 m/s" << std::endl;
            std::cerr << std::endl;
            std::cerr << "  " << Color::CYAN << "Fix in:" << Color::RESET << " " << filename << std::endl;
            std::cerr << std::endl;
            exit(1);
        }
        if (drydepositionVelocity[i] > 1.0f) {
            std::cerr << std::endl << Color::YELLOW << Color::BOLD << "[WARNING] "
                      << Color::RESET << "Very large deposition velocity for nuclide " << (i+1)
                      << ": " << drydepositionVelocity[i] << " m/s" << std::endl;
            std::cerr << "  Typical deposition velocities are < 0.1 m/s." << std::endl;
            std::cerr << "  Such high values suggest rapid gravitational settling (large particles)." << std::endl;
            std::cerr << std::endl;
        }
    }

    std::cout << Color::GREEN << "done" << Color::RESET << std::endl;

    // Print loaded configuration
    std::cout << Color::BOLD << "Nuclide Configuration" << Color::RESET << std::endl;
    std::cout << "  File               : " << filename << std::endl;
    std::cout << "  Nuclides loaded    : " << Color::BOLD << nuclide_count << Color::RESET << std::endl;

    // Print first nuclide as example
    if (nuclide_count > 0) {
        std::cout << "  Decay constant     : " << decayConstants[0] << " s⁻¹" << std::endl;
        std::cout << "  Deposition velocity: " << drydepositionVelocity[0] << " m/s" << std::endl;
    }
}

/******************************************************************************
 * @brief Load advanced system configuration from advanced.conf
 *
 * @details Validates grid dimensions and coordinate system parameters against
 *          compile-time constants. Provides early warning if config file
 *          dimensions differ from code constants.
 *
 *          Checks:
 *          - gfs_dimX vs Constants::dimX_GFS
 *          - gfs_dimY vs Constants::dimY_GFS
 *          - gfs_dimZ vs Constants::dimZ_GFS
 *
 * @pre input/advanced.conf must exist
 * @post Grid dimensions validated and reported
 *
 * @note Code always uses Constants namespace values (compile-time)
 * @note Dimension mismatch generates warning, not error
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::loadAdvancedConfig() {
    ConfigReader adv_config;

    if (!adv_config.loadConfig("input/advanced.conf")) {
        std::cerr << Color::RED << "[ERROR]" << Color::RESET
                  << " Failed to load input/advanced.conf" << std::endl;
        std::cerr << "This file contains advanced system parameters." << std::endl;
        std::cerr << "If missing, create it using util/generate_config_templates.py" << std::endl;
        exit(1);
    }

    // Load grid dimensions for validation (NO DEFAULTS - must match compile-time constants)
    std::string dimX_str = adv_config.getString("gfs_dimX", "");
    std::string dimY_str = adv_config.getString("gfs_dimY", "");
    std::string dimZ_str = adv_config.getString("gfs_dimZ", "");

    // These parameters are informational/validation only
    // Code always uses Constants namespace values (compile-time)
    // So we allow defaults equal to the compile-time constants
    int cfg_gfs_dimX = dimX_str.empty() ? Constants::dimX_GFS : std::stoi(dimX_str);
    int cfg_gfs_dimY = dimY_str.empty() ? Constants::dimY_GFS : std::stoi(dimY_str);
    int cfg_gfs_dimZ = dimZ_str.empty() ? Constants::dimZ_GFS : std::stoi(dimZ_str);

    // Validate grid dimensions
    bool dimensions_match = (cfg_gfs_dimX == Constants::dimX_GFS) &&
                           (cfg_gfs_dimY == Constants::dimY_GFS) &&
                           (cfg_gfs_dimZ == Constants::dimZ_GFS);

    // Output validation result
    std::cout << Color::BOLD << "Advanced Configuration" << Color::RESET << std::endl;
    std::cout << "  Data paths: " << (isGFS ? "GFS" : "LDAPS") << std::endl;

    if (dimensions_match) {
        std::cout << "  Grid dimensions: " << Color::GREEN << "validated" << Color::RESET << std::endl;
    } else {
        std::cout << "  Grid dimensions: " << Color::YELLOW << "MISMATCH" << Color::RESET << std::endl;
        std::cout << Color::YELLOW << "  Warning: " << Color::RESET
                  << "Config dimensions differ from code constants" << std::endl;
        std::cout << "  Code will use Constants namespace values (compile-time)" << std::endl;
    }
}
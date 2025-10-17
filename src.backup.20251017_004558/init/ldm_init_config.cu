/**
 * @file ldm_init_config.cu
 * @brief Implementation of configuration loading and initialization functions
 */

#include "../core/ldm.cuh"
#include "../physics/ldm_nuclides.cuh"
#include "colors.h"

void LDM::loadSimulationConfiguration(){

    if (!g_config.loadConfig("input/setting.txt")) {
        std::cerr << "Failed to load configuration file" << std::endl;
        exit(1);
    }

    FILE* sourceFile;

    time_end = g_config.getFloat("Time_end(s)", 64800.0f);
    dt = g_config.getFloat("dt(s)", 10.0f);
    freq_output = g_config.getInt("Plot_output_freq", 10);
    nop = g_config.getInt("Total_number_of_particle", 10000);
    isRural = g_config.getInt("Rural/Urban", 1);
    isPG = g_config.getInt("Pasquill-Gifford/Briggs-McElroy-Pooler", 1);
    isGFS = g_config.getInt("Data", 1);

    // Load terminal output settings
    g_sim.fixedScrollOutput = g_config.getInt("fixed_scroll_output", 1);

    // Load physics model settings
    g_turb_switch = g_config.getInt("turbulence_model", 0);
    g_drydep = g_config.getInt("dry_deposition_model", 0);
    g_wetdep = g_config.getInt("wet_deposition_model", 0);
    g_raddecay = g_config.getInt("radioactive_decay_model", 1);
    
    // Print physics model status
    std::cout << Color::BOLD << "Physics Models" << Color::RESET << std::endl;
    std::cout << "  TURB=" << g_turb_switch
              << ", DRYDEP=" << g_drydep
              << ", WETDEP=" << g_wetdep
              << ", RADDECAY=" << g_raddecay << std::endl;
    
    // Clean output directory before simulation
    cleanOutputDirectory();
    
    //loadRadionuclideData();

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
    
    // Initialize CRAM system with dynamic dt
    if (initialize_cram_system("cram/A60.csv")) {
        // Compute exp(-A*dt) matrix using the dt from configuration
    } else {
        std::cerr << "Warning: CRAM system initialization failed, using traditional decay" << std::endl;
    }

    std::string source_file_path = g_config.getString("input_base_path", "./input/") + "source.txt";
    sourceFile = fopen(source_file_path.c_str(), "r");

    if (!sourceFile){
        std::cerr << "Failed to open source.txt" << std::endl;
        exit(1);
    }

    char buffer[256];
    while (fgets(buffer, sizeof(buffer), sourceFile)){
        if (buffer[0] == '#') continue;
    
        // SOURCE coordinates
        if (strstr(buffer, "[SOURCE]")) {
            while (fgets(buffer, sizeof(buffer), sourceFile) && !strstr(buffer, "[SOURCE_TERM]")) {
                if (buffer[0] == '#') continue;
    
                Source src;
                sscanf(buffer, "%f %f %f", &src.lon, &src.lat, &src.height);
                sources.push_back(src);
            }
            sources.pop_back();
        }
    
        // SOURCE_TERM values
        if (strstr(buffer, "[SOURCE_TERM]")){
            while (fgets(buffer, sizeof(buffer), sourceFile) && !strstr(buffer, "[RELEASE_CASES]")) {
                if (buffer[0] == '#') continue;
    
                int srcnum;
                float decay, depvel;
                sscanf(buffer, "%d %f %f", &srcnum, &decay, &depvel);
                decayConstants.push_back(decay);
                drydepositionVelocity.push_back(depvel);
            }
            decayConstants.pop_back();
            drydepositionVelocity.pop_back();
        }
    
        // RELEASE_CASES
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

    //nop = floor(nop/(sources.size()*decayConstants.size()))*sources.size()*decayConstants.size();

    // Note: All configuration values are now passed via KernelScalars struct to kernels
    // No need for cudaMemcpyToSymbol since __constant__ symbols have been removed

}
void LDM::cleanOutputDirectory() {
    std::cout << "Cleaning output directory... " << std::flush;

    // Remove all files in output directory
    #ifdef _WIN32
        system("del /Q output\\*.* 2>nul");
    #else
        system("rm -f output/*.vtk 2>/dev/null");
        system("rm -f output/*.csv 2>/dev/null");
        system("rm -f output/*.txt 2>/dev/null");
    #endif

    std::cout << Color::GREEN << "" << Color::RESET << std::endl;
}

void LDM::loadEKISettings() {
    std::cout << "\n" << Color::CYAN << "[SYSTEM] " << Color::RESET
              << "Loading EKI settings... " << std::flush;
    
    FILE* ekiFile = fopen("input/eki_settings.txt", "r");
    if (!ekiFile) {
        std::cerr << "Failed to open input/eki_settings.txt" << std::endl;
        exit(1);
    }
    
    char buffer[256];
    
    // Initialize global EKI config with default values
    g_eki.mode = true;  // Set to true since this function is called when EKI mode is ON
    g_eki.time_interval = 15.0f;
    g_eki.time_unit = "minutes";
    g_eki.num_receptors = 0;
    g_eki.prior_mode = "constant";
    g_eki.prior_constant = 1.5e+8f;
    g_eki.ensemble_size = 50;
    g_eki.noise_level = 0.01f;
    
    // Clear vectors
    g_eki.receptor_locations.clear();
    g_eki.true_emissions.clear();
    g_eki.prior_emissions.clear();
    
    bool reading_receptor_matrix = false;
    bool reading_true_emissions = false;
    bool reading_prior_emissions = false;
    
    while (fgets(buffer, sizeof(buffer), ekiFile)) {
        // Skip comments and empty lines
        if (buffer[0] == '#' || buffer[0] == '\n' || buffer[0] == '\r') {
            continue;
        }
        
        // Parsing debug disabled - working correctly
        
        // Check for section headers
        if (strstr(buffer, "RECEPTOR_LOCATIONS_MATRIX=")) {
            reading_receptor_matrix = true;
            reading_true_emissions = false;
            reading_prior_emissions = false;
            continue;
        }
        
        if (strstr(buffer, "TRUE_EMISSION_SERIES=")) {
            reading_receptor_matrix = false;
            reading_true_emissions = true;
            reading_prior_emissions = false;
            continue;
        }
        
        if (strstr(buffer, "PRIOR_EMISSION_SERIES=")) {
            reading_receptor_matrix = false;
            reading_true_emissions = false;
            reading_prior_emissions = true;
            continue;
        }
        
        // Check if this line is a key-value pair (contains '=')
        // If so, reset section flags and parse as key-value
        if (strchr(buffer, '=') != nullptr) {
            reading_receptor_matrix = false;
            reading_true_emissions = false;
            reading_prior_emissions = false;
            // Section flags reset for key-value parsing
        }
        
        // Parse based on current section
        if (reading_receptor_matrix) {
            float lat, lon;
            if (sscanf(buffer, "%f %f", &lat, &lon) == 2) {
                g_eki.receptor_locations.push_back(std::make_pair(lat, lon));
            }
        }
        else if (reading_true_emissions) {
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
        
        // Parse key-value pairs (now that section flags are reset if needed)
        if (strchr(buffer, '=') != nullptr) {
            
            // Parse key-value pairs
            if (strstr(buffer, "EKI_TIME_INTERVAL=")) {
                sscanf(buffer, "EKI_TIME_INTERVAL=%f", &g_eki.time_interval);
            }
            else if (strstr(buffer, "EKI_TIME_UNIT=")) {
                char unit[32];
                sscanf(buffer, "EKI_TIME_UNIT=%s", unit);
                g_eki.time_unit = std::string(unit);
            }
            else if (strstr(buffer, "NUM_RECEPTORS=")) {
                sscanf(buffer, "NUM_RECEPTORS=%d", &g_eki.num_receptors);
            }
            else if (strstr(buffer, "RECEPTOR_CAPTURE_RADIUS=")) {
                sscanf(buffer, "RECEPTOR_CAPTURE_RADIUS=%f", &g_eki.receptor_capture_radius);
            }
            else if (strstr(buffer, "PRIOR_EMISSION_MODE=")) {
                char mode[32];
                sscanf(buffer, "PRIOR_EMISSION_MODE=%s", mode);
                g_eki.prior_mode = std::string(mode);
            }
            else if (strstr(buffer, "PRIOR_EMISSION_CONSTANT=")) {
                sscanf(buffer, "PRIOR_EMISSION_CONSTANT=%f", &g_eki.prior_constant);
            }
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
            else if (strstr(buffer, "EKI_GPU_FORWARD=")) {
                char opt[32];
                sscanf(buffer, "EKI_GPU_FORWARD=%s", opt);
                g_eki.gpu_forward = std::string(opt);
            }
            else if (strstr(buffer, "EKI_GPU_INVERSE=")) {
                char opt[32];
                sscanf(buffer, "EKI_GPU_INVERSE=%s", opt);
                g_eki.gpu_inverse = std::string(opt);
            }
            else if (strstr(buffer, "EKI_NUM_GPU=")) {
                sscanf(buffer, "EKI_NUM_GPU=%d", &g_eki.num_gpu);
            }
            else if (strstr(buffer, "EKI_TIME_DAYS=")) {
                sscanf(buffer, "EKI_TIME_DAYS=%f", &g_eki.time_days);
            }
            else if (strstr(buffer, "EKI_INVERSE_TIME_INTERVAL=")) {
                sscanf(buffer, "EKI_INVERSE_TIME_INTERVAL=%f", &g_eki.inverse_time_interval);
            }
            else if (strstr(buffer, "EKI_RECEPTOR_ERROR=")) {
                sscanf(buffer, "EKI_RECEPTOR_ERROR=%f", &g_eki.receptor_error);
            }
            else if (strstr(buffer, "EKI_RECEPTOR_MDA=")) {
                sscanf(buffer, "EKI_RECEPTOR_MDA=%f", &g_eki.receptor_mda);
            }
            else if (strstr(buffer, "EKI_SOURCE_LOCATION=")) {
                char loc[32];
                sscanf(buffer, "EKI_SOURCE_LOCATION=%s", loc);
                g_eki.source_location = std::string(loc);
            }
            else if (strstr(buffer, "EKI_NUM_SOURCE=")) {
                sscanf(buffer, "EKI_NUM_SOURCE=%d", &g_eki.num_source);
            }
            else if (strstr(buffer, "MEMORY_DOCTOR_MODE=")) {
                char mode[32];
                sscanf(buffer, "MEMORY_DOCTOR_MODE=%s", mode);
                g_eki.memory_doctor_mode = (strcmp(mode, "On") == 0 || strcmp(mode, "on") == 0 ||
                                           strcmp(mode, "ON") == 0 || strcmp(mode, "1") == 0);
            }
        }
        
        // Reset section flags only when encountering empty line (not newline characters at end)
        // Note: buffer[0] should not be '\r' or '\n' for normal config lines
        // This check should only trigger for truly empty lines, not lines with Windows line endings
    }
    
    fclose(ekiFile);

    std::cout << Color::GREEN << "" << Color::RESET << std::endl;

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

// ================== GRID RECEPTOR DEBUG MODE FUNCTIONS ==================

void LDM::initializeGridReceptors(int grid_count_param, float grid_spacing_param) {
    // Store grid parameters
    grid_count = grid_count_param;
    grid_spacing = grid_spacing_param;
    grid_receptor_total = (2 * grid_count + 1) * (2 * grid_count + 1);

    std::cout << Color::CYAN << "[SYSTEM] " << Color::RESET
              << "Initializing " << Color::BOLD << grid_receptor_total << Color::RESET
              << " grid receptors (" << (2*grid_count+1) << "×" << (2*grid_count+1)
              << ", spacing=" << grid_spacing << "°)" << std::endl;

    // Source location (from setting.txt default)
    float source_lat = 37.0f;
    float source_lon = 141.0f;

    // Prepare host arrays for receptor locations
    std::vector<float> host_lats(grid_receptor_total);
    std::vector<float> host_lons(grid_receptor_total);

    // Generate grid receptor locations
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

// ================== MODERNIZED CONFIG LOADING FUNCTIONS (Phase 1) ==================
// These functions implement the new input file structure as described in
// docs/INPUT_MODERNIZATION_PLAN.md

/**
 * @brief Load simulation parameters from modernized simulation.conf file
 */
void LDM::loadSimulationConfig() {
    std::cout << Color::CYAN << "[CONFIG] " << Color::RESET
              << "Loading simulation.conf... " << std::flush;

    // Load configuration file
    if (!g_config.loadConfig("input/simulation.conf")) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET
                  << "Failed to load input/simulation.conf" << std::endl;
        std::cerr << "         Please ensure the file exists and is readable." << std::endl;
        exit(1);
    }

    // ========== TEMPORAL SETTINGS ==========
    time_end = g_config.getFloat("time_end", 21600.0f);  // Default: 6 hours
    dt = g_config.getFloat("time_step", 100.0f);         // Default: 100 seconds
    freq_output = g_config.getInt("vtk_output_frequency", 1);  // Default: every timestep

    // ========== PARTICLE SETTINGS ==========
    nop = g_config.getInt("total_particles", 10000);     // Default: 10,000 particles

    // ========== ATMOSPHERIC CONDITIONS ==========
    isRural = g_config.getInt("rural_conditions", 1);    // Default: Rural (1)
    isPG = g_config.getInt("use_pasquill_gifford", 1);   // Default: Pasquill-Gifford (1)

    // ========== METEOROLOGICAL DATA SOURCE ==========
    isGFS = g_config.getInt("use_gfs_data", 1);          // Default: GFS (1)

    // ========== TERMINAL OUTPUT ==========
    g_sim.fixedScrollOutput = g_config.getInt("fixed_scroll_output", 1);  // Default: Enabled (1)

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
}

/**
 * @brief Load physics model configuration from physics.conf
 */
void LDM::loadPhysicsConfig() {
    std::cout << Color::CYAN << "[SYSTEM] " << Color::RESET
              << "Loading physics configuration... " << std::flush;

    // Load physics.conf using ConfigReader
    ConfigReader physics_config;
    if (!physics_config.loadConfig("input/physics.conf")) {
        std::cerr << Color::RED << "[ERROR]" << Color::RESET
                  << " Failed to load input/physics.conf" << std::endl;
        exit(1);
    }

    // Parse physics model switches
    g_turb_switch = physics_config.getInt("turbulence_model", 0);
    g_drydep = physics_config.getInt("dry_deposition_model", 0);
    g_wetdep = physics_config.getInt("wet_deposition_model", 0);
    g_raddecay = physics_config.getInt("radioactive_decay_model", 1);

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

/**
 * @brief Load source locations from source.conf file
 */
void LDM::loadSourceConfig() {
    std::cout << Color::CYAN << "[SYSTEM] " << Color::RESET
              << "Loading source locations... " << std::flush;

    // Construct file path
    std::string source_file_path = g_config.getString("input_base_path", "./input/") + "source.conf";

    FILE* sourceFile = fopen(source_file_path.c_str(), "r");

    if (!sourceFile) {
        std::cerr << Color::RED << "Failed to open " << source_file_path << Color::RESET << std::endl;
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

        // Parse source location: LON LAT HEIGHT
        Source src;
        int parsed = sscanf(buffer, "%f %f %f", &src.lon, &src.lat, &src.height);

        if (parsed == 3) {
            sources.push_back(src);
        } else {
            std::cerr << Color::YELLOW << "\n[WARNING] " << Color::RESET
                      << "Skipping invalid line " << line_number
                      << " in source.conf: " << buffer;
        }
    }

    fclose(sourceFile);

    // Validation: at least one source must be defined
    if (sources.empty()) {
        std::cerr << Color::RED << "\n[ERROR] " << Color::RESET
                  << "No valid sources found in source.conf" << std::endl;
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

/**
 * @brief Load nuclide configuration from nuclides.conf
 */
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
                decayConstants.push_back(fabs(decay_const));  // Ensure positive
                drydepositionVelocity.push_back(1.0f);  // Default deposition velocity
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
        std::cerr << std::endl << Color::RED << "[ERROR] " << Color::RESET
                  << "No nuclides loaded from " << filename << std::endl;
        exit(1);
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

/**
 * @brief Load advanced system configuration from advanced.conf
 */
void LDM::loadAdvancedConfig() {
    ConfigReader adv_config;

    if (!adv_config.loadConfig("input/advanced.conf")) {
        std::cerr << Color::RED << "[ERROR]" << Color::RESET
                  << " Failed to load input/advanced.conf" << std::endl;
        std::cerr << "This file contains advanced system parameters." << std::endl;
        std::cerr << "If missing, create it using util/generate_config_templates.py" << std::endl;
        exit(1);
    }

    // Load grid dimensions for validation
    int cfg_gfs_dimX = adv_config.getInt("gfs_dimX", Constants::dimX_GFS);
    int cfg_gfs_dimY = adv_config.getInt("gfs_dimY", Constants::dimY_GFS);
    int cfg_gfs_dimZ = adv_config.getInt("gfs_dimZ", Constants::dimZ_GFS);

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
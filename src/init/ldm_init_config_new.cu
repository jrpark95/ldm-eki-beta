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

/******************************************************************************
 * @file ldm.cu
 * @brief Implementation of LDM class constructors, destructors, and helpers
 *
 * This file contains:
 * - Global variable definitions for simulation configuration
 * - LDM class constructor/destructor implementations
 * - EKIMeteorologicalData cleanup functions
 * - Grid configuration helper functions
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/

#include "ldm.cuh"

// ===========================================================================
// Global Variable Definitions (declared in ldm.cuh as extern)
// ===========================================================================
// Note: These must be defined AFTER including ldm.cuh but before LDM class
//       implementation to avoid multiple definition errors in the linker.
//       Each .cu file that includes ldm.cuh sees the extern declarations,
//       but only ldm.cu provides the actual storage allocation.

SimulationConfig g_sim;            // Simulation parameters (time, particles, etc.)
MPIConfig g_mpi;                   // MPI/species configuration (legacy compatibility)
EKIConfig g_eki;                   // EKI algorithm parameters and receptor settings
ConfigReader g_config;             // Generic configuration file reader
EKIMeteorologicalData g_eki_meteo; // Preloaded meteorology for EKI iterations
std::vector<float> flex_hgt;       // Vertical height levels (host copy)

// Log-only output stream (initialized in main_eki.cu, nullptr otherwise)
std::ostream* g_logonly = nullptr;

// ===========================================================================
// LDM Class Constructor and Destructor
// ===========================================================================

/******************************************************************************
 * @brief Default constructor for LDM class
 *
 * Initializes all member variables to safe default values. GPU memory pointers
 * are set to nullptr to enable safe destruction and conditional deallocation.
 *
 * Member Initialization:
 * - Ensemble mode flags: false/0
 * - GPU pointers: nullptr (allocated later in allocateGPUMemory())
 * - Observation counters: 0
 * - Grid receptor mode: disabled
 *
 * @note Actual GPU memory allocation happens in allocateGPUMemory()
 * @note Configuration loading happens via load*Config() methods
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
LDM::LDM()
    : is_ensemble_mode(false)
    , ensemble_size(0)
    , ensemble_num_states(0)
    , enable_vtk_output(false)
    , current_eki_iteration(0)
    , max_eki_iterations(0)
    , is_grid_receptor_mode(false)
    , grid_count(0)
    , grid_spacing(0.0f)
    , grid_receptor_total(0)
    , eki_observation_count(0)
    , d_part(nullptr)
    , d_grid_receptor_lats(nullptr)
    , d_grid_receptor_lons(nullptr)
    , d_grid_receptor_dose(nullptr)
    , d_grid_receptor_particle_count(nullptr)
    , d_receptor_lats(nullptr)
    , d_receptor_lons(nullptr)
    , d_receptor_dose(nullptr)
    , d_receptor_particle_count(nullptr)
    , d_ensemble_dose(nullptr)
    , d_ensemble_particle_count(nullptr)
    , d_T_matrix(nullptr)
    , d_flex_hgt(nullptr)
    , device_meteorological_data_pres(nullptr)
    , device_meteorological_data_unis(nullptr)
    , device_meteorological_data_etas(nullptr)
    , device_meteorological_flex_pres0(nullptr)
    , device_meteorological_flex_pres1(nullptr)
    , device_meteorological_flex_pres2(nullptr)
    , device_meteorological_flex_unis0(nullptr)
    , device_meteorological_flex_unis1(nullptr)
    , device_meteorological_flex_unis2(nullptr)
{
    // Constructor body empty - all initialization in member initializer list
}

/******************************************************************************
 * @brief Destructor for LDM class
 *
 * Releases GPU memory allocated for CRAM decay matrix and height levels.
 * Other GPU resources (particles, meteorology, observations) are cleaned up
 * by their respective cleanup functions (cleanupEKIObservationSystem, etc.)
 *
 * @note Uses conditional deallocation (if ptr != nullptr) for safety
 * @note cudaFree(nullptr) is safe but checking explicitly is clearer
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
LDM::~LDM() {
    // Free CRAM T matrix GPU memory
    if (d_T_matrix != nullptr) {
        cudaFree(d_T_matrix);
        d_T_matrix = nullptr;
    }

    // Free height data GPU memory
    if (d_flex_hgt != nullptr) {
        cudaFree(d_flex_hgt);
        d_flex_hgt = nullptr;
    }

    // Other resources cleaned up by explicit cleanup functions:
    // - Particles: allocateGPUMemory() manages lifecycle
    // - Observations: cleanupEKIObservationSystem()
    // - Meteorology: cleanupEKIMeteorologicalData()
}

// ===========================================================================
// EKIMeteorologicalData Implementation
// ===========================================================================

/******************************************************************************
 * @brief Destructor for EKIMeteorologicalData
 *
 * Automatically releases all allocated memory (host and GPU) when the
 * meteorological data cache goes out of scope. Calls cleanup() internally.
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
EKIMeteorologicalData::~EKIMeteorologicalData() {
    cleanup();
}

/******************************************************************************
 * @brief Release all preloaded meteorological data from memory
 *
 * This function performs comprehensive cleanup of the EKI meteorology cache:
 * 1. Deallocate host memory arrays (FlexPres, FlexUnis, height data)
 * 2. Deallocate GPU memory for all timesteps
 * 3. Free pointer arrays on GPU
 * 4. Reset metadata (sizes, counts, initialization flag)
 *
 * Memory Release Order (Important):
 * - First: Retrieve GPU pointers to individual timestep arrays
 * - Second: Free individual timestep arrays on GPU
 * - Third: Free pointer arrays themselves on GPU
 * - Fourth: Free host memory and clear vectors
 *
 * Error Handling:
 * - Uses try/catch to prevent cleanup failures from crashing program
 * - Continues cleanup even if individual cudaFree calls fail
 * - Logs errors to stderr with color-coded messages
 *
 * @note Safe to call multiple times (checks is_initialized flag)
 * @note Called automatically by destructor
 * @note After cleanup, preloadAllEKIMeteorologicalData() can be called again
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void EKIMeteorologicalData::cleanup() {
    if (!is_initialized) {
        return;
    }

    try {
        // Clean up host memory
        for (size_t i = 0; i < host_flex_pres_data.size(); i++) {
            if (host_flex_pres_data[i] != nullptr) {
                delete[] host_flex_pres_data[i];
                host_flex_pres_data[i] = nullptr;
            }
        }
        for (size_t i = 0; i < host_flex_unis_data.size(); i++) {
            if (host_flex_unis_data[i] != nullptr) {
                delete[] host_flex_unis_data[i];
                host_flex_unis_data[i] = nullptr;
            }
        }
        host_flex_pres_data.clear();
        host_flex_unis_data.clear();
        host_flex_hgt_data.clear();

        // Clean up GPU memory (order is important!)

        // First release individual GPU memory blocks
        if (device_flex_pres_data && num_time_steps > 0) {
            // Get pointer array from GPU
            std::vector<FlexPres*> temp_pres_ptrs(num_time_steps);
            cudaError_t err = cudaMemcpy(temp_pres_ptrs.data(), device_flex_pres_data,
                                       num_time_steps * sizeof(FlexPres*), cudaMemcpyDeviceToHost);
            if (err == cudaSuccess) {
                for (int i = 0; i < num_time_steps; i++) {
                    if (temp_pres_ptrs[i] != nullptr) {
                        cudaFree(temp_pres_ptrs[i]);
                    }
                }
            }
            cudaFree(device_flex_pres_data);
            device_flex_pres_data = nullptr;
        }

        if (device_flex_unis_data && num_time_steps > 0) {
            std::vector<FlexUnis*> temp_unis_ptrs(num_time_steps);
            cudaError_t err = cudaMemcpy(temp_unis_ptrs.data(), device_flex_unis_data,
                                       num_time_steps * sizeof(FlexUnis*), cudaMemcpyDeviceToHost);
            if (err == cudaSuccess) {
                for (int i = 0; i < num_time_steps; i++) {
                    if (temp_unis_ptrs[i] != nullptr) {
                        cudaFree(temp_unis_ptrs[i]);
                    }
                }
            }
            cudaFree(device_flex_unis_data);
            device_flex_unis_data = nullptr;
        }

        if (device_flex_hgt_data && num_time_steps > 0) {
            std::vector<float*> temp_hgt_ptrs(num_time_steps);
            cudaError_t err = cudaMemcpy(temp_hgt_ptrs.data(), device_flex_hgt_data,
                                       num_time_steps * sizeof(float*), cudaMemcpyDeviceToHost);
            if (err == cudaSuccess) {
                for (int i = 0; i < num_time_steps; i++) {
                    if (temp_hgt_ptrs[i] != nullptr) {
                        cudaFree(temp_hgt_ptrs[i]);
                    }
                }
            }
            cudaFree(device_flex_hgt_data);
            device_flex_hgt_data = nullptr;
        }

        // Reset metadata
        num_time_steps = 0;
        pres_data_size = 0;
        unis_data_size = 0;
        hgt_data_size = 0;
        is_initialized = false;

        // Clean up existing LDM GPU memory slots
        if (ldm_pres0_slot) {
            cudaFree(ldm_pres0_slot);
            ldm_pres0_slot = nullptr;
        }
        if (ldm_unis0_slot) {
            cudaFree(ldm_unis0_slot);
            ldm_unis0_slot = nullptr;
        }
        if (ldm_pres1_slot) {
            cudaFree(ldm_pres1_slot);
            ldm_pres1_slot = nullptr;
        }
        if (ldm_unis1_slot) {
            cudaFree(ldm_unis1_slot);
            ldm_unis1_slot = nullptr;
        }

    } catch (const std::exception& e) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Exception during memory cleanup: " << e.what() << std::endl;
        is_initialized = false;
    } catch (...) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Unknown exception during memory cleanup" << std::endl;
        is_initialized = false;
    }
}

// ===========================================================================
// Grid Configuration Helper Function
// ===========================================================================

/******************************************************************************
 * @brief Load grid configuration from source configuration file
 *
 * Parses the [GRID_CONFIG] section in source.conf (or legacy source.txt) to
 * extract parameters for spatial discretization of the simulation domain.
 * Used for grid-based output and concentration field calculations.
 *
 * Configuration Format:
 * [GRID_CONFIG]
 * start_lat: 35.0
 * start_lon: 129.0
 * end_lat: 37.0
 * end_lon: 131.0
 * lat_step: 0.5
 * lon_step: 0.5
 *
 * Grid Dimensions:
 * - Latitude range: [start_lat, end_lat] degrees
 * - Longitude range: [start_lon, end_lon] degrees
 * - Grid cells: (end_lat - start_lat) / lat_step × (end_lon - start_lon) / lon_step
 *
 * @return GridConfig structure with parsed values
 *
 * @pre Configuration file must exist (source.conf or source.txt)
 * @pre [GRID_CONFIG] section must be present with all required parameters
 * @pre All values must be valid floats within geographic ranges
 *
 * @post Program exits with error if file not found or values invalid
 *
 * @note Fail-fast: Exits immediately on error (no default values)
 * @note Ignores lines starting with '#' (comments)
 * @note Tries source.conf first, then source.txt for backward compatibility
 *
 * @see GridConfig struct definition in ldm.cuh
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
GridConfig loadGridConfig() {
    GridConfig config;

    // Try modern config file first, fallback to legacy
    std::string source_file_path;
    std::ifstream file;

    // Try source.conf first
    source_file_path = g_config.getString("input_base_path", "./input/") + "source.conf";
    file.open(source_file_path);

    // Fallback to source.txt
    if (!file.is_open()) {
        source_file_path = g_config.getString("input_base_path", "./input/") + "source.txt";
        file.open(source_file_path);
    }

    if (!file.is_open()) {
        std::cerr << "\n" << Color::RED << Color::BOLD << "[ERROR] " << Color::RESET
                  << "Failed to open source configuration file for grid config" << std::endl;
        std::cerr << "  Tried: input/source.conf, input/source.txt" << std::endl;
        std::cerr << "\n  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
        std::cerr << "    Grid configuration requires source location file" << std::endl;
        std::cerr << "\n  " << Color::CYAN << "Solution:" << Color::RESET << std::endl;
        std::cerr << "    Ensure input/source.conf exists with [GRID_CONFIG] section" << std::endl;
        std::cerr << "\n  " << Color::GREEN << "Example format:" << Color::RESET << std::endl;
        std::cerr << "    [GRID_CONFIG]" << std::endl;
        std::cerr << "    start_lat: 36.0" << std::endl;
        std::cerr << "    start_lon: 140.0" << std::endl;
        std::cerr << "    end_lat: 37.0" << std::endl;
        std::cerr << "    end_lon: 141.0" << std::endl;
        std::cerr << "    lat_step: 0.5" << std::endl;
        std::cerr << "    lon_step: 0.5" << std::endl;
        std::cerr << std::endl;
        exit(1);
    }

    std::string line;
    bool in_grid_section = false;

    while (std::getline(file, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;

        // Detect [GRID_CONFIG] section
        if (line == "[GRID_CONFIG]") {
            in_grid_section = true;
            continue;
        }

        // Exit grid section when encountering another section
        if (line.find('[') != std::string::npos) {
            in_grid_section = false;
            continue;
        }

        // Parse key-value pairs within grid section
        if (in_grid_section) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);

                // Trim key and value
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);

                try {
                    if (key == "start_lat") config.start_lat = std::stof(value);
                    else if (key == "start_lon") config.start_lon = std::stof(value);
                    else if (key == "end_lat") config.end_lat = std::stof(value);
                    else if (key == "end_lon") config.end_lon = std::stof(value);
                    else if (key == "lat_step") config.lat_step = std::stof(value);
                    else if (key == "lon_step") config.lon_step = std::stof(value);
                } catch (const std::exception& e) {
                    std::cerr << "\n" << Color::RED << Color::BOLD << "[INPUT ERROR] " << Color::RESET
                              << "Invalid value for " << Color::BOLD << key << Color::RESET << ": " << value << std::endl;
                    std::cerr << "\n  " << Color::YELLOW << "Problem:" << Color::RESET << std::endl;
                    std::cerr << "    Cannot parse value as floating-point number" << std::endl;
                    std::cerr << "    Exception: " << e.what() << std::endl;
                    std::cerr << "\n  " << Color::CYAN << "Required format:" << Color::RESET << std::endl;
                    std::cerr << "    " << key << ": <number>" << std::endl;
                    std::cerr << "\n  " << Color::GREEN << "Examples:" << Color::RESET << std::endl;
                    std::cerr << "    start_lat: 35.0" << std::endl;
                    std::cerr << "    lon_step: 0.5" << std::endl;
                    std::cerr << "\n  " << Color::CYAN << "Fix in:" << Color::RESET << " " << source_file_path << std::endl;
                    std::cerr << std::endl;
                    exit(1);
                }
            }
        }
    }

    return config;
}

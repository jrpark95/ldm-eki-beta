/**
 * @file ldm.cu
 * @brief Implementation of LDM class member functions and helper functions
 */

#include "ldm.cuh"

// ============================================================================
// Global Variable Definitions (declared in ldm.cuh as extern)
// ============================================================================
// Note: These must be defined AFTER including ldm.cuh but before LDM class implementation

SimulationConfig g_sim;
MPIConfig g_mpi;
EKIConfig g_eki;
ConfigReader g_config;
EKIMeteorologicalData g_eki_meteo;
std::vector<float> flex_hgt;

// Log-only output stream (initialized in main_eki.cu)
std::ostream* g_logonly = nullptr;

// ============================================================================
// LDM Class Constructor and Destructor
// ============================================================================

LDM::LDM()
    : is_ensemble_mode(false)
    , ensemble_size(0)
    , ensemble_num_states(0)
    , enable_vtk_output(false)
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
    // Constructor implementation - initialize member variables
}

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

    // Cleanup other resources
}

// ============================================================================
// EKIMeteorologicalData Implementation
// ============================================================================

EKIMeteorologicalData::~EKIMeteorologicalData() {
    cleanup();
}

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

// ============================================================================
// Grid Configuration Helper Function
// ============================================================================

GridConfig loadGridConfig() {
    GridConfig config;

    std::string source_file_path = g_config.getString("input_base_path", "./input/") + "source.txt";
    std::ifstream file(source_file_path);

    if (!file.is_open()) {
        std::cerr << "Warning: Could not open source.txt for grid config, using defaults" << std::endl;
        return config;
    }

    std::string line;
    bool in_grid_section = false;

    while (std::getline(file, line)) {
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (line.empty() || line[0] == '#') continue;

        if (line == "[GRID_CONFIG]") {
            in_grid_section = true;
            continue;
        }

        if (line.find('[') != std::string::npos) {
            in_grid_section = false;
            continue;
        }

        if (in_grid_section) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);

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
                    std::cerr << "Warning: Invalid value for " << key << ": " << value << std::endl;
                }
            }
        }
    }

    return config;
}

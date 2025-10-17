/**
 * @file ldm_plot_utils.cu
 * @brief Implementation of visualization utility and validation functions
 */

#include "../core/ldm.cuh"
#include "../physics/ldm_nuclides.cuh"
#include "colors.h"

// ============================================================================
// VTK Utility Functions
// ============================================================================

int LDM::countActiveParticles(){
    int count = 0;
    for(int i = 0; i < nop; ++i) if(part[i].flag == 1) count++;
    return count;
}

void LDM::swapByteOrder(float& value){
    char* valuePtr = reinterpret_cast<char*>(&value);
    std::swap(valuePtr[0], valuePtr[3]);
    std::swap(valuePtr[1], valuePtr[2]);
}

void LDM::swapByteOrder(int& value){
    char* valuePtr = reinterpret_cast<char*>(&value);
    std::swap(valuePtr[0], valuePtr[3]);
    std::swap(valuePtr[1], valuePtr[2]);
}

// ============================================================================
// Validation and Logging Functions
// ============================================================================

void LDM::log_first_particle_concentrations(int timestep, float currentTime) {
    // Copy particles from device to host
    cudaMemcpy(part.data(), d_part, nop * sizeof(LDMpart), cudaMemcpyDeviceToHost);

    // Ensure validation directory exists
    #ifdef _WIN32
        _mkdir("validation");
    #else
        mkdir("validation", 0777);
    #endif

    // Create or append to CSV file
    static bool first_write = true;
    std::string filename = "validation/first_particle_concentrations.csv";
    std::ofstream csvFile;

    if (first_write) {
        csvFile.open(filename, std::ios::out);
        // Write header with dynamic nuclide names
        csvFile << "timestep,time(s),total_conc";
        NuclideConfig* nucConfig = NuclideConfig::getInstance();
        int num_nuclides = nucConfig->getNumNuclides();
        for (int i = 0; i < num_nuclides; i++) {
            csvFile << "," << nucConfig->getNuclideName(i);
        }
        csvFile << std::endl;
        first_write = false;
    } else {
        csvFile.open(filename, std::ios::app);
    }

    // Find first active particle
    bool found_active = false;
    for (size_t idx = 0; idx < part.size(); idx++) {
        const auto& p = part[idx];
        if (p.flag) {
            found_active = true;
            NuclideConfig* nucConfig = NuclideConfig::getInstance();
            int num_nuclides = nucConfig->getNumNuclides();
            csvFile << timestep << "," << currentTime << "," << p.conc;
            for (int i = 0; i < num_nuclides; i++) {
                csvFile << "," << p.concentrations[i];
            }
            csvFile << std::endl;
            break; // Only log the first active particle
        }
    }
    if (!found_active) {
        if (!part.empty()) {
            const auto& p = part[0];
            NuclideConfig* nucConfig = NuclideConfig::getInstance();
            int num_nuclides = nucConfig->getNumNuclides();
            csvFile << timestep << "," << currentTime << "," << p.conc;
            for (int i = 0; i < num_nuclides; i++) {
                csvFile << "," << p.concentrations[i];
            }
            csvFile << std::endl;
        }
    }

    csvFile.close();
}

void LDM::log_all_particles_nuclide_ratios(int timestep, float currentTime) {
    // Copy particles from device to host
    cudaMemcpy(part.data(), d_part, nop * sizeof(LDMpart), cudaMemcpyDeviceToHost);

    // Ensure validation directory exists
    #ifdef _WIN32
        _mkdir("validation");
    #else
        mkdir("validation", 0777);
    #endif

    // Create or append to CSV file
    static bool first_write = true;
    std::string filename = "validation/all_particles_nuclide_ratios.csv";
    std::ofstream csvFile;

    if (first_write) {
        csvFile.open(filename, std::ios::out);
        // Write header
        csvFile << "timestep,time(s),active_particles,total_conc";
        for (int i = 0; i < MAX_NUCLIDES; i++) {
            csvFile << ",total_Q_" << i << ",ratio_Q_" << i;
        }
        csvFile << std::endl;
        first_write = false;
    } else {
        csvFile.open(filename, std::ios::app);
    }

    // Calculate totals for all active particles
    float total_concentrations[MAX_NUCLIDES] = {0.0f};
    float total_conc = 0.0f;
    int active_particles = 0;

    for (const auto& p : part) {
        if (p.flag) {
            active_particles++;
            total_conc += p.conc;
            for (int i = 0; i < MAX_NUCLIDES; i++) {
                total_concentrations[i] += p.concentrations[i];
            }
        }
    }

    // Write data
    csvFile << timestep << "," << currentTime << "," << active_particles << "," << total_conc;
    for (int i = 0; i < MAX_NUCLIDES; i++) {
        float ratio = (total_conc > 0) ? (total_concentrations[i] / total_conc) : 0.0f;
        csvFile << "," << total_concentrations[i] << "," << ratio;
    }
    csvFile << std::endl;

    csvFile.close();
}

void LDM::log_first_particle_cram_detail(int timestep, float currentTime, float dt_used) {
    // Copy particles from device to host
    cudaMemcpy(part.data(), d_part, nop * sizeof(LDMpart), cudaMemcpyDeviceToHost);

    // Ensure validation directory exists
    #ifdef _WIN32
        _mkdir("validation");
    #else
        mkdir("validation", 0777);
    #endif

    // Create or append to detailed CSV file
    static bool first_write = true;
    std::string filename = "validation/first_particle_cram_detail.csv";
    std::ofstream csvFile;

    if (first_write) {
        csvFile.open(filename, std::ios::out);
        // Write detailed header with decay information
        csvFile << "timestep,time(s),dt(s),particle_age(s)";

        NuclideConfig* nucConfig = NuclideConfig::getInstance();
        int num_nuclides = nucConfig->getNumNuclides();

        for (int i = 0; i < num_nuclides; i++) {
            std::string name = nucConfig->getNuclideName(i);
            float half_life = nucConfig->getHalfLife(i);
            csvFile << "," << name << "_conc," << name << "_half_life," << name << "_decay_factor";
        }
        csvFile << ",total_mass,mass_conservation_check" << std::endl;
        first_write = false;
    } else {
        csvFile.open(filename, std::ios::app);
    }

    // Find first active particle and log detailed information
    for (const auto& p : part) {
        if (p.flag) {
            NuclideConfig* nucConfig = NuclideConfig::getInstance();
            int num_nuclides = nucConfig->getNumNuclides();

            csvFile << timestep << "," << currentTime << "," << dt_used << "," << p.age;

            float total_mass = 0.0f;
            for (int i = 0; i < num_nuclides; i++) {
                float half_life = nucConfig->getHalfLife(i);
                float decay_constant = log(2.0f) / (half_life * 3600.0f); // Convert hours to seconds
                float decay_factor = exp(-decay_constant * dt_used);

                csvFile << "," << p.concentrations[i] << "," << half_life << "," << decay_factor;
                total_mass += p.concentrations[i];
            }

            // Mass conservation check (should decrease due to decay)
            static float initial_mass = -1.0f;
            if (initial_mass < 0) initial_mass = total_mass;
            float conservation_ratio = total_mass / initial_mass;

            csvFile << "," << total_mass << "," << conservation_ratio << std::endl;
            break; // Only log the first active particle
        }
    }

    csvFile.close();
}

void LDM::log_first_particle_decay_analysis(int timestep, float currentTime) {
    // Copy particles from device to host
    cudaMemcpy(part.data(), d_part, nop * sizeof(LDMpart), cudaMemcpyDeviceToHost);

    // Ensure validation directory exists
    #ifdef _WIN32
        _mkdir("validation");
    #else
        mkdir("validation", 0777);
    #endif

    static bool first_write = true;
    std::string filename = "validation/first_particle_decay_analysis.csv";
    std::ofstream csvFile;

    if (first_write) {
        csvFile.open(filename, std::ios::out);
        csvFile << "timestep,time(s),nuclide_name,concentration,half_life_hours,decay_constant_per_sec,theoretical_concentration,relative_error" << std::endl;
        first_write = false;
    } else {
        csvFile.open(filename, std::ios::app);
    }

    // Find first active particle
    for (const auto& p : part) {
        if (p.flag) {
            NuclideConfig* nucConfig = NuclideConfig::getInstance();
            int num_nuclides = nucConfig->getNumNuclides();

            // Analyze each nuclide
            for (int i = 0; i < num_nuclides; i++) {
                std::string name = nucConfig->getNuclideName(i);
                float half_life = nucConfig->getHalfLife(i);
                float decay_constant = log(2.0f) / (half_life * 3600.0f);

                // Calculate theoretical concentration (starting from initial 0.1)
                float theoretical_conc = 0.1f * exp(-decay_constant * p.age);
                float relative_error = (p.concentrations[i] - theoretical_conc) / theoretical_conc * 100.0f;

                csvFile << timestep << "," << currentTime << "," << name << ","
                       << p.concentrations[i] << "," << half_life << "," << decay_constant << ","
                       << theoretical_conc << "," << relative_error << std::endl;
            }
            break; // Only analyze the first active particle
        }
    }

    csvFile.close();
}

// ============================================================================
// Validation Data Export Functions
// ============================================================================

void LDM::exportValidationData(int timestep, float currentTime) {
    // Create validation folder
    static bool validation_dir_created = false;
    if (!validation_dir_created) {
        #ifdef _WIN32
            _mkdir("validation");
        #else
            mkdir("validation", 0777);
        #endif
        validation_dir_created = true;
    }

    // Copy particles from device to host
    cudaMemcpy(part.data(), d_part, nop * sizeof(LDMpart), cudaMemcpyDeviceToHost);

    // Export grid data only at key timesteps (save disk space)
    if (timestep % 50 == 0 || timestep <= 10 || timestep >= 710) {
        exportConcentrationGrid(timestep, currentTime);
    }

    // Export nuclide totals at every timestep
    exportNuclideTotal(timestep, currentTime);

    if (timestep % 100 == 0) {
        std::cout << "[VALIDATION] Exported reference data for timestep " << timestep << std::endl;
    }
}

void LDM::exportConcentrationGrid(int timestep, float currentTime) {
    // Fukushima region setup (139-143°E, 36-39°N, 0-2000m)
    const float min_lon = 139.0f, max_lon = 143.0f;
    const float min_lat = 36.0f, max_lat = 39.0f;
    const float min_alt = 0.0f, max_alt = 2000.0f;
    const int grid_x = 100, grid_y = 100, grid_z = 20;

    const float dx = (max_lon - min_lon) / grid_x;
    const float dy = (max_lat - min_lat) / grid_y;
    const float dz = (max_alt - min_alt) / grid_z;

    // Initialize grid
    std::vector<std::vector<std::vector<float>>> concentration_grid(
        grid_x, std::vector<std::vector<float>>(grid_y, std::vector<float>(grid_z, 0.0f)));
    std::vector<std::vector<std::vector<int>>> count_grid(
        grid_x, std::vector<std::vector<int>>(grid_y, std::vector<int>(grid_z, 0)));

    // Map active particles to grid
    for (const auto& p : part) {
        if (!p.flag) continue;

        // Convert GFS coordinates to geographic coordinates
        float lon = -179.0f + p.x * 0.5f;
        float lat = -90.0f + p.y * 0.5f;
        float alt = p.z;

        // Check grid bounds
        if (lon < min_lon || lon >= max_lon || lat < min_lat || lat >= max_lat ||
            alt < min_alt || alt >= max_alt) continue;

        // Calculate grid indices
        int ix = static_cast<int>((lon - min_lon) / dx);
        int iy = static_cast<int>((lat - min_lat) / dy);
        int iz = static_cast<int>((alt - min_alt) / dz);

        // Boundary check
        if (ix >= 0 && ix < grid_x && iy >= 0 && iy < grid_y && iz >= 0 && iz < grid_z) {
            concentration_grid[ix][iy][iz] += p.conc;
            count_grid[ix][iy][iz]++;
        }
    }

    // Save grid data to CSV file
    std::ostringstream filename;
    filename << "validation/concentration_grid_" << std::setfill('0') << std::setw(5) << timestep << ".csv";

    std::ofstream csvFile(filename.str());
    csvFile << "x_index,y_index,z_index,lon,lat,alt,concentration,particle_count" << std::endl;

    for (int ix = 0; ix < grid_x; ix++) {
        for (int iy = 0; iy < grid_y; iy++) {
            for (int iz = 0; iz < grid_z; iz++) {
                if (concentration_grid[ix][iy][iz] > 0 || count_grid[ix][iy][iz] > 0) {
                    float lon = min_lon + (ix + 0.5f) * dx;
                    float lat = min_lat + (iy + 0.5f) * dy;
                    float alt = min_alt + (iz + 0.5f) * dz;

                    csvFile << ix << "," << iy << "," << iz << ","
                           << lon << "," << lat << "," << alt << ","
                           << concentration_grid[ix][iy][iz] << ","
                           << count_grid[ix][iy][iz] << std::endl;
                }
            }
        }
    }
    csvFile.close();
}

void LDM::exportNuclideTotal(int timestep, float currentTime) {
    static bool first_write = true;
    std::string filename = "validation/nuclide_totals.csv";
    std::ofstream csvFile;

    if (first_write) {
        csvFile.open(filename, std::ios::out);
        csvFile << "timestep,time(s),active_particles,total_conc";
        NuclideConfig* nucConfig = NuclideConfig::getInstance();
        int num_nuclides = nucConfig->getNumNuclides();
        for (int i = 0; i < num_nuclides; i++) {
            csvFile << ",total_" << nucConfig->getNuclideName(i);
        }
        csvFile << std::endl;
        first_write = false;
    } else {
        csvFile.open(filename, std::ios::app);
    }

    // Calculate total concentrations per nuclide
    std::vector<float> total_concentrations(MAX_NUCLIDES, 0.0f);
    float total_conc = 0.0f;
    int active_particles = 0;

    for (const auto& p : part) {
        if (p.flag) {
            active_particles++;
            total_conc += p.conc;
            NuclideConfig* nucConfig = NuclideConfig::getInstance();
            int num_nuclides = nucConfig->getNumNuclides();
            for (int i = 0; i < num_nuclides; i++) {
                total_concentrations[i] += p.concentrations[i];
            }
        }
    }

    // Write data
    csvFile << timestep << "," << currentTime << "," << active_particles << "," << total_conc;
    NuclideConfig* nucConfig = NuclideConfig::getInstance();
    int num_nuclides = nucConfig->getNumNuclides();
    for (int i = 0; i < num_nuclides; i++) {
        csvFile << "," << total_concentrations[i];
    }
    csvFile << std::endl;

    csvFile.close();
}

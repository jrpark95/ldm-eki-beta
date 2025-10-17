/**
 * @file ldm_plot_utils.cu
 * @brief Implementation of visualization utility and validation functions
 * @author Juryong Park
 * @date 2025
 *
 * @details Implements utility functions for VTK file generation and
 *          validation/logging functions for model verification.
 */

#include "../core/ldm.cuh"
#include "../physics/ldm_nuclides.cuh"
#include "colors.h"

// ============================================================================
// VTK Utility Functions
// ============================================================================

/**
 * @implementation countActiveParticles
 *
 * Simple linear scan through host particle array counting active flags.
 * This is called once per VTK output (not performance-critical).
 */
int LDM::countActiveParticles(){
    int count = 0;
    for(int i = 0; i < nop; ++i) if(part[i].flag == 1) count++;
    return count;
}

/**
 * @implementation swapByteOrder (float)
 *
 * Byte-swapping for IEEE 754 single-precision float (32 bits = 4 bytes).
 * Converts between little-endian (x86) and big-endian (VTK requirement).
 *
 * @byte_layout
 * Before: [LSB] [B1] [B2] [MSB]  (little-endian)
 * After:  [MSB] [B2] [B1] [LSB]  (big-endian)
 */
void LDM::swapByteOrder(float& value){
    char* valuePtr = reinterpret_cast<char*>(&value);
    std::swap(valuePtr[0], valuePtr[3]);  // Swap outermost bytes
    std::swap(valuePtr[1], valuePtr[2]);  // Swap innermost bytes
}

/**
 * @implementation swapByteOrder (int)
 *
 * Byte-swapping for signed 32-bit integer.
 * Same logic as float version (4 bytes, 2 swaps).
 */
void LDM::swapByteOrder(int& value){
    char* valuePtr = reinterpret_cast<char*>(&value);
    std::swap(valuePtr[0], valuePtr[3]);
    std::swap(valuePtr[1], valuePtr[2]);
}

// ============================================================================
// Validation and Logging Functions
// ============================================================================
//
// These functions are used for development, debugging, and model verification.
// They export time-series data to CSV files for external analysis in Python,
// R, or spreadsheet software.
//
// Performance impact: ~5-10% slowdown due to GPU→Host copies and file I/O
// Recommendation: Disable in production runs by commenting out calls

/**
 * @implementation log_first_particle_concentrations
 *
 * Tracks a single representative particle over time to observe transport
 * and decay behavior. Useful for debugging concentration changes and
 * verifying that decay chains are working correctly.
 *
 * @output_format CSV with columns:
 * - timestep: Simulation timestep number
 * - time(s): Simulation time in seconds
 * - total_conc: Total concentration (sum of all nuclides)
 * - <nuclide_name>: Concentration for each nuclide (dynamic columns)
 */
void LDM::log_first_particle_concentrations(int timestep, float currentTime) {
    // Copy particles from GPU to host
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
        // Write header with dynamic nuclide names from configuration
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

/**
 * @implementation log_all_particles_nuclide_ratios
 *
 * Aggregates concentrations across all active particles and calculates
 * relative ratios for each nuclide. Useful for:
 * - Verifying mass conservation
 * - Checking decay chain evolution
 * - Comparing with reference implementations
 *
 * @output_format CSV with columns:
 * - timestep, time(s), active_particles, total_conc
 * - For each nuclide i: total_Q_i (absolute), ratio_Q_i (relative)
 *
 * @invariants
 * - Sum of ratios should equal 1.0 (within numerical precision)
 * - Total concentration should decrease monotonically (decay)
 */
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

/**
 * @implementation log_first_particle_cram_detail
 *
 * Logs detailed CRAM decay information for a single particle to verify
 * numerical accuracy of the Chebyshev Rational Approximation Method.
 * Records pre/post decay data with half-lives and computed decay factors.
 *
 * @output_format CSV with columns:
 * - timestep, time(s), dt(s), particle_age(s)
 * - For each nuclide: <name>_conc, <name>_half_life, <name>_decay_factor
 * - total_mass, mass_conservation_check
 *
 * @algorithm
 * For each nuclide:
 *   1. Retrieve half-life from configuration
 *   2. Calculate decay constant: λ = ln(2) / T_{1/2}
 *   3. Calculate decay factor: exp(-λ * dt)
 *   4. Compare with CRAM-computed concentration
 *
 * @validation_metric mass_conservation_check = current_mass / initial_mass
 *   - Should decrease monotonically (due to decay)
 *   - Should not increase (would indicate non-physical behavior)
 */
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

/**
 * @implementation log_first_particle_decay_analysis
 *
 * Compares CRAM-computed concentrations with theoretical exponential decay
 * to quantify numerical accuracy. This function validates that the CRAM
 * implementation correctly handles radioactive decay chains.
 *
 * @output_format CSV with columns:
 * - timestep, time(s), nuclide_name
 * - concentration: CRAM-computed value
 * - half_life_hours: Nuclide half-life [hours]
 * - decay_constant_per_sec: λ [s⁻¹]
 * - theoretical_concentration: C₀ * exp(-λt)
 * - relative_error: (CRAM - theory) / theory × 100%
 *
 * @algorithm
 * For each nuclide in first active particle:
 *   1. Retrieve half-life from configuration
 *   2. Calculate decay constant: λ = ln(2) / T_{1/2}
 *   3. Calculate theoretical: C_theory = C₀ * exp(-λ * age)
 *   4. Calculate relative error: ε = (C_CRAM - C_theory) / C_theory
 *
 * @accuracy_expectations
 * - Relative error should be < 1% for CRAM48
 * - Errors increase for very short half-lives (stiff ODEs)
 * - Errors increase for long simulation times
 *
 * @note Initial concentration C₀ = 0.1 (hardcoded)
 */
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

/**
 * @implementation exportValidationData
 *
 * Orchestrates export of validation datasets for model verification.
 * This master function coordinates grid and nuclide total exports.
 *
 * @export_strategy
 * - Grid data: Selected timesteps only (save disk space)
 *   - Every 50 timesteps
 *   - First 10 timesteps (initial plume development)
 *   - Last 10 timesteps (final state)
 * - Nuclide totals: Every timestep (lightweight CSV)
 *
 * @disk_usage
 * - Full export: ~1-5 GB for 720 timesteps
 * - Selective export: ~100-500 MB
 */
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

/**
 * @implementation exportConcentrationGrid
 *
 * Converts unstructured particle cloud to structured 3D grid for spatial
 * analysis. Grid is designed for Fukushima region but can be adjusted.
 *
 * @algorithm
 * 1. Define grid domain and resolution
 * 2. Initialize concentration and count grids (100×100×20)
 * 3. For each active particle:
 *    - Convert GFS coords to geographic (lon, lat, alt)
 *    - Check if particle is within grid bounds
 *    - Calculate grid cell indices
 *    - Accumulate concentration and count
 * 4. Write sparse CSV (only non-empty cells)
 *
 * @memory_usage ~16 MB for grid storage (200,000 cells × 2 arrays × 4 bytes)
 */
void LDM::exportConcentrationGrid(int timestep, float currentTime) {
    // Define grid domain (Fukushima region)
    const float min_lon = 139.0f, max_lon = 143.0f;  // 4° longitude span
    const float min_lat = 36.0f, max_lat = 39.0f;    // 3° latitude span
    const float min_alt = 0.0f, max_alt = 2000.0f;   // 2 km altitude
    const int grid_x = 100, grid_y = 100, grid_z = 20;

    // Calculate cell dimensions
    const float dx = (max_lon - min_lon) / grid_x;  // ~0.04° per cell
    const float dy = (max_lat - min_lat) / grid_y;  // ~0.03° per cell
    const float dz = (max_alt - min_alt) / grid_z;  // 100 m per cell

    // Initialize 3D grids
    std::vector<std::vector<std::vector<float>>> concentration_grid(
        grid_x, std::vector<std::vector<float>>(grid_y, std::vector<float>(grid_z, 0.0f)));
    std::vector<std::vector<std::vector<int>>> count_grid(
        grid_x, std::vector<std::vector<int>>(grid_y, std::vector<int>(grid_z, 0)));

    // Map active particles to grid
    for (const auto& p : part) {
        if (!p.flag) continue;

        // Convert GFS grid coordinates to geographic coordinates
        float lon = -179.0f + p.x * 0.5f;  // Longitude
        float lat = -90.0f + p.y * 0.5f;   // Latitude
        float alt = p.z;                   // Altitude [m]

        // Check if particle is within grid bounds
        if (lon < min_lon || lon >= max_lon || lat < min_lat || lat >= max_lat ||
            alt < min_alt || alt >= max_alt) continue;

        // Calculate grid cell indices
        int ix = static_cast<int>((lon - min_lon) / dx);
        int iy = static_cast<int>((lat - min_lat) / dy);
        int iz = static_cast<int>((alt - min_alt) / dz);

        // Boundary check (redundant but safe)
        if (ix >= 0 && ix < grid_x && iy >= 0 && iy < grid_y && iz >= 0 && iz < grid_z) {
            concentration_grid[ix][iy][iz] += p.conc;
            count_grid[ix][iy][iz]++;
        }
    }

    // Save grid data to CSV file (sparse format)
    std::ostringstream filename;
    filename << "validation/concentration_grid_" << std::setfill('0') << std::setw(5) << timestep << ".csv";

    std::ofstream csvFile(filename.str());
    csvFile << "x_index,y_index,z_index,lon,lat,alt,concentration,particle_count" << std::endl;

    // Write only non-empty cells (sparse format saves disk space)
    for (int ix = 0; ix < grid_x; ix++) {
        for (int iy = 0; iy < grid_y; iy++) {
            for (int iz = 0; iz < grid_z; iz++) {
                if (concentration_grid[ix][iy][iz] > 0 || count_grid[ix][iy][iz] > 0) {
                    // Calculate cell center coordinates
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

/**
 * @implementation exportNuclideTotal
 *
 * Aggregates and exports total concentration per nuclide for mass
 * conservation validation and decay chain verification.
 *
 * @output_format CSV with columns:
 * - timestep, time(s), active_particles, total_conc
 * - total_<nuclide_name> for each configured nuclide
 *
 * @algorithm
 * 1. Initialize accumulator for each nuclide
 * 2. For each active particle:
 *    - Add particle's total concentration
 *    - Add each nuclide's concentration to respective accumulator
 * 3. Write timestep row to CSV
 *
 * @validation_checks
 * - Total mass should decrease monotonically (radioactive decay)
 * - Sum of nuclide totals should equal total_conc
 * - Active particle count tracks dispersion progress
 *
 * @performance O(N × M) where N = active particles, M = nuclides
 */
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

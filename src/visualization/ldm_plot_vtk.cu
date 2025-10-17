/**
 * @file ldm_plot_vtk.cu
 * @brief Implementation of VTK output functions
 * @author Juryong Park
 * @date 2025
 *
 * @details Implements VTK Legacy format (Version 4.0) output for particle
 *          visualization. The implementation handles both single-mode and
 *          ensemble-mode simulations with the following features:
 *
 *          - Binary encoding with big-endian byte order (VTK standard)
 *          - Coordinate system conversion (GFS grid → Geographic)
 *          - Active particle filtering (flag == 1)
 *          - Parallel I/O for ensemble mode (OpenMP)
 *
 * @note All binary data is byte-swapped to big-endian on x86 systems
 */

#include "../core/ldm.cuh"
#include "../physics/ldm_nuclides.cuh"
#include "colors.h"
#include <omp.h>

// ============================================================================
// Single-Mode VTK Output
// ============================================================================

/**
 * @implementation outputParticlesBinaryMPI
 *
 * @algorithm
 * 1. Copy particle data from GPU to host memory
 * 2. Count active particles (flag == 1)
 * 3. Create output directory (output/plot_vtk_prior/)
 * 4. Open binary VTK file for writing
 * 5. Write ASCII header (VTK version, format, dataset type)
 * 6. Write binary POINTS section:
 *    - Convert GFS grid coordinates to geographic (lon, lat, alt)
 *    - Apply altitude scaling (z/3000) for better visualization
 *    - Byte-swap to big-endian
 * 7. Write binary POINT_DATA section:
 *    - Q: Particle concentration [Bq/m³]
 *    - time_idx: Emission time index
 * 8. Close file
 *
 * @optimization Active particle filtering avoids writing inactive particles
 * @coordinate_transform lon = -179.0 + x*0.5, lat = -90.0 + y*0.5, alt = z/3000
 */
void LDM::outputParticlesBinaryMPI(int timestep){

    // Step 1: Copy particle data from GPU to host
    cudaMemcpy(part.data(), d_part, nop * sizeof(LDMpart), cudaMemcpyDeviceToHost);

    // Step 2: Count active particles for file header
    int part_num = countActiveParticles();

    // Step 3: Create output directory and filename
    std::ostringstream filenameStream;
    std::string path = "output/plot_vtk_prior";

    #ifdef _WIN32
        _mkdir(path.c_str());
        filenameStream << path << "\\" << "plot_" << std::setfill('0')
                       << std::setw(5) << timestep << ".vtk";
    #else
        mkdir(path.c_str(), 0777);
        filenameStream << path << "/" << "plot_" << std::setfill('0')
                       << std::setw(5) << timestep << ".vtk";
    #endif
    std::string filename = filenameStream.str();

    // Step 4: Open binary VTK file for writing
    std::ofstream vtkFile(filename, std::ios::binary);

    if (!vtkFile.is_open()){
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return;
    }

    // Step 5: Write ASCII header section
    vtkFile << "# vtk DataFile Version 4.0\n";
    vtkFile << "particle data\n";
    vtkFile << "BINARY\n";
    vtkFile << "DATASET POLYDATA\n";

    // Step 6: Write binary POINTS section (geometry)
    vtkFile << "POINTS " << part_num << " float\n";
    float zsum = 0.0;
    for (int i = 0; i < nop; ++i){
        if(!part[i].flag) continue;  // Skip inactive particles

        // Convert GFS grid coordinates to geographic coordinates
        // GFS grid: x ∈ [0, 719] (0.5° resolution), y ∈ [0, 359]
        // Geographic: lon ∈ [-179°, +180°], lat ∈ [-90°, +90°]
        float x = -179.0 + part[i].x * 0.5;  // Longitude
        float y = -90.0 + part[i].y * 0.5;   // Latitude
        float z = part[i].z / 3000.0;        // Scaled altitude for visualization

        zsum += part[i].z;  // Accumulate for statistics (unused)

        // VTK binary format requires big-endian byte order
        swapByteOrder(x);
        swapByteOrder(y);
        swapByteOrder(z);

        vtkFile.write(reinterpret_cast<char*>(&x), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&y), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&z), sizeof(float));
    }

    // Step 7: Write binary POINT_DATA section (attributes)
    vtkFile << "POINT_DATA " << part_num << "\n";

    // Optional fields (commented out for cleaner output):
    // - u_wind, v_wind, w_wind: Wind velocity components
    // - virtual_dist: Virtual distance for parameterizations
    // - I131_concentration: Specific nuclide tracking
    // These can be uncommented for detailed debugging/analysis

    // Attribute 1: Q (Particle concentration)
    vtkFile << "SCALARS Q float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nop; ++i){
        if(!part[i].flag) continue;
        float vval = part[i].conc;
        swapByteOrder(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    // Attribute 2: time_idx (Emission time index)
    vtkFile << "SCALARS time_idx int 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nop; ++i){
        if(!part[i].flag) continue;
        int vval = part[i].timeidx;
        swapByteOrder(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(int));
    }

    // Step 8: Close file
    vtkFile.close();
}

// ============================================================================
// Ensemble-Mode VTK Output with Parallel I/O
// ============================================================================

/**
 * @implementation outputParticlesBinaryMPI_ens
 *
 * @algorithm
 * 1. Set OpenMP threads for parallel I/O (50 threads)
 * 2. Copy all ensemble particles from GPU to host
 * 3. Validate ensemble mode and selected ensembles
 * 4. Pre-filter particles by ensemble_id (selected ensembles only)
 * 5. OpenMP parallel loop over selected ensembles:
 *    a. Create ensemble-specific VTK file (ens_XXX_timestep_XXXXX.vtk)
 *    b. Write ASCII header with ensemble number
 *    c. Write binary POINTS (coordinate conversion)
 *    d. Write binary POINT_DATA (Q, time_idx)
 * 6. All files written concurrently
 *
 * @optimization Pre-filtering (step 4) avoids scanning all particles per ensemble
 * @parallelization OpenMP dynamic scheduling balances load across threads
 * @scaling ~10-20x speedup vs sequential for 100 ensembles
 */
void LDM::outputParticlesBinaryMPI_ens(int timestep){

    // Step 1: Configure OpenMP for optimal parallel I/O
    // Balance between parallelism and system resources
    // (50 threads chosen empirically for 56-thread system)
    omp_set_num_threads(50);

    // Step 2: Copy all ensemble particles from GPU to host
    size_t total_particles = part.size();

    if (total_particles == 0) {
        std::cerr << Color::RED << Color::BOLD << "[ERROR] " << Color::RESET
                  << "Particle vector is empty in ensemble output\n";
        return;
    }

    cudaMemcpy(part.data(), d_part, total_particles * sizeof(LDMpart), cudaMemcpyDeviceToHost);

    // Step 3: Create output directory
    std::string path = "output/plot_vtk_ens";

    #ifdef _WIN32
        _mkdir(path.c_str());
    #else
        mkdir(path.c_str(), 0777);
    #endif

    // Validate ensemble mode
    if (!is_ensemble_mode) {
        std::cerr << Color::YELLOW << "[WARNING] " << Color::RESET
                  << "Ensemble output called but not in ensemble mode\n";
        return;
    }

    // Validate selected ensembles
    if (selected_ensemble_ids.empty()) {
        std::cerr << Color::YELLOW << "[WARNING] " << Color::RESET
                  << "No ensembles selected for output, skipping VTK\n";
        return;
    }

    // Step 4: Pre-filter particles by ensemble_id (OPTIMIZATION)
    // Only selected ensembles (typically 3) instead of all (typically 100)
    std::vector<std::vector<int>> ensemble_particle_indices(ensemble_size);
    for (int i = 0; i < total_particles; ++i) {
        if (part[i].flag && part[i].ensemble_id >= 0 && part[i].ensemble_id < ensemble_size) {
            // Check if this ensemble is selected for output
            bool is_selected = false;
            for (int selected_id : selected_ensemble_ids) {
                if (part[i].ensemble_id == selected_id) {
                    is_selected = true;
                    break;
                }
            }
            if (is_selected) {
                ensemble_particle_indices[part[i].ensemble_id].push_back(i);
            }
        }
    }

    // Step 5: Parallel loop over selected ensembles
    // Dynamic scheduling balances load (ensembles may have different particle counts)
    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < selected_ensemble_ids.size(); idx++) {
        int ens = selected_ensemble_ids[idx];
        const auto& particle_indices = ensemble_particle_indices[ens];
        int part_num = particle_indices.size();

        if (part_num == 0) continue;  // Skip empty ensembles

        // Create ensemble-specific filename
        std::ostringstream filenameStream;
        #ifdef _WIN32
            filenameStream << path << "\\ens_" << std::setfill('0') << std::setw(3) << ens
                          << "_timestep_" << std::setw(5) << timestep << ".vtk";
        #else
            filenameStream << path << "/ens_" << std::setfill('0') << std::setw(3) << ens
                          << "_timestep_" << std::setw(5) << timestep << ".vtk";
        #endif
        std::string filename = filenameStream.str();

        std::ofstream vtkFile(filename, std::ios::binary);

        if (!vtkFile.is_open()){
            std::cerr << "Cannot open file for writing: " << filename << std::endl;
            continue;
        }

        // Write ASCII header with ensemble identifier
        vtkFile << "# vtk DataFile Version 4.0\n";
        vtkFile << "Ensemble " << ens << " particle data\n";
        vtkFile << "BINARY\n";
        vtkFile << "DATASET POLYDATA\n";

        // Write binary POINTS section
        vtkFile << "POINTS " << part_num << " float\n";
        float zsum = 0.0;
        for (int idx : particle_indices){
            // Coordinate conversion (same as single mode)
            float x = -179.0 + part[idx].x * 0.5;
            float y = -90.0 + part[idx].y * 0.5;
            float z = part[idx].z / 3000.0;
            zsum += part[idx].z;

            swapByteOrder(x);
            swapByteOrder(y);
            swapByteOrder(z);

            vtkFile.write(reinterpret_cast<char*>(&x), sizeof(float));
            vtkFile.write(reinterpret_cast<char*>(&y), sizeof(float));
            vtkFile.write(reinterpret_cast<char*>(&z), sizeof(float));
        }

        // Write binary POINT_DATA section
        vtkFile << "POINT_DATA " << part_num << "\n";

        // Attribute 1: Q (Particle concentration)
        vtkFile << "SCALARS Q float 1\n";
        vtkFile << "LOOKUP_TABLE default\n";
        for (int idx : particle_indices){
            float vval = part[idx].conc;
            swapByteOrder(vval);
            vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
        }

        // Attribute 2: time_idx (Emission time index)
        vtkFile << "SCALARS time_idx int 1\n";
        vtkFile << "LOOKUP_TABLE default\n";
        for (int idx : particle_indices){
            int vval = part[idx].timeidx;
            swapByteOrder(vval);
            vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(int));
        }

        vtkFile.close();
    } // End of parallel ensemble loop
}

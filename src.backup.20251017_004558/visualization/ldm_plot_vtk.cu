/**
 * @file ldm_plot_vtk.cu
 * @brief Implementation of VTK output functions
 */

#include "../core/ldm.cuh"
#include "../physics/ldm_nuclides.cuh"
#include "colors.h"
#include <omp.h>

void LDM::outputParticlesBinaryMPI(int timestep){

    cudaMemcpy(part.data(), d_part, nop * sizeof(LDMpart), cudaMemcpyDeviceToHost);
    int numa = 0;
    int numb = 0;

    int part_num = countActiveParticles();

    // Debug output disabled for release

    std::ostringstream filenameStream;
    std::string path;

    // Create output directory for VTK files - NEW STRUCTURE
    path = "output/plot_vtk_prior";

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

    std::ofstream vtkFile(filename, std::ios::binary);

    if (!vtkFile.is_open()){
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return;
    }

    vtkFile << "# vtk DataFile Version 4.0\n";
    vtkFile << "particle data\n";
    vtkFile << "BINARY\n";
    vtkFile << "DATASET POLYDATA\n";

    vtkFile << "POINTS " << part_num << " float\n";
    float zsum = 0.0;
    for (int i = 0; i < nop; ++i){
        if(!part[i].flag) continue;
        // float x = part[i].x;
        // float y = part[i].y;
        // float z = part[i].z/3000.0;

        float x = -179.0 + part[i].x*0.5;
        float y = -90.0 + part[i].y*0.5;
        float z = part[i].z/3000.0;
        zsum += part[i].z;

        swapByteOrder(x);
        swapByteOrder(y);
        swapByteOrder(z);

        vtkFile.write(reinterpret_cast<char*>(&x), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&y), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&z), sizeof(float));
    }
    // __Z[timestep-1]=zsum/static_cast<float>(part_num);


    vtkFile << "POINT_DATA " << part_num << "\n";
    // vtkFile << "SCALARS u_wind float 1\n";
    // vtkFile << "LOOKUP_TABLE default\n";
    // for (int i = 0; i < nop; ++i){
    //     if(!part[i].flag) continue;
    //     float vval = part[i].u_wind;
    //     swapByteOrder(vval);
    //     vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    // }

    // vtkFile << "SCALARS v_wind float 1\n";
    // vtkFile << "LOOKUP_TABLE default\n";
    // for (int i = 0; i < nop; ++i){
    //     if(!part[i].flag) continue;
    //     float vval = part[i].v_wind;
    //     swapByteOrder(vval);
    //     vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    // }

    // vtkFile << "SCALARS w_wind float 1\n";
    // vtkFile << "LOOKUP_TABLE default\n";
    // for (int i = 0; i < nop; ++i){
    //     if(!part[i].flag) continue;
    //     float vval = part[i].w_wind;
    //     swapByteOrder(vval);
    //     vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    // }

    // vtkFile << "SCALARS virtual_dist float 1\n";
    // vtkFile << "LOOKUP_TABLE default\n";
    // for (int i = 0; i < nop; ++i){
    //     if(!part[i].flag) continue;
    //     float vval = part[i].virtual_distance;
    //     swapByteOrder(vval);
    //     vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    // }

    vtkFile << "SCALARS Q float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nop; ++i){
        if(!part[i].flag) continue;
        float vval = part[i].conc;
        swapByteOrder(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS time_idx int 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nop; ++i){
        if(!part[i].flag) continue;
        int vval = part[i].timeidx;
        swapByteOrder(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(int));
    }

    // // Output representative nuclide concentration (I-131, index 32)
    // const int representative_nuclide = 32;  // I-131 is at index 32 in the 60-nuclide configuration
    // vtkFile << "SCALARS I131_concentration float 1\n";
    // vtkFile << "LOOKUP_TABLE default\n";
    // for (int i = 0; i < nop; ++i){
    //     if(!part[i].flag) continue;
    //     float vval = part[i].concentrations[representative_nuclide];
    //     swapByteOrder(vval);
    //     vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    // }


    vtkFile.close();
}

void LDM::outputParticlesBinaryMPI_ens(int timestep){

    // Set OpenMP threads for optimal performance (28 physical cores × 2 = 56 threads)
    // Use 48 threads to leave headroom for system and GPU tasks
    omp_set_num_threads(50);

    // VTK configuration info is now printed in main_eki.cu before simulation starts
    // to avoid interfering with progress bar display

    // Ensemble mode: use actual particle count from part vector
    // This was set during initializeParticlesEKI_AllEnsembles()
    size_t total_particles = part.size();

    if (total_particles == 0) {
        std::cerr << Color::RED << Color::BOLD << "[ERROR] " << Color::RESET
                  << "Particle vector is empty in ensemble output\n";
        return;
    }

    // Copy all particles from GPU
    cudaMemcpy(part.data(), d_part, total_particles * sizeof(LDMpart), cudaMemcpyDeviceToHost);

    // Create output_ens directory - NEW STRUCTURE
    std::string path = "output/plot_vtk_ens";

    #ifdef _WIN32
        _mkdir(path.c_str());
    #else
        mkdir(path.c_str(), 0777);
    #endif

    // Group particles by ensemble_id and write separate files
    if (!is_ensemble_mode) {
        std::cerr << Color::YELLOW << "[WARNING] " << Color::RESET
                  << "Ensemble output called but not in ensemble mode\n";
        return;
    }

    // Check if selected ensembles are initialized
    if (selected_ensemble_ids.empty()) {
        std::cerr << Color::YELLOW << "[WARNING] " << Color::RESET
                  << "No ensembles selected for output, skipping VTK\n";
        return;
    }

    // Pre-filter particles only for selected ensembles (3 ensembles instead of all 100)
    std::vector<std::vector<int>> ensemble_particle_indices(ensemble_size);
    for (int i = 0; i < total_particles; ++i) {
        if (part[i].flag && part[i].ensemble_id >= 0 && part[i].ensemble_id < ensemble_size) {
            // Only filter if this ensemble is selected for output
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

    // Process only selected ensembles (3 ensembles) with OpenMP parallelization
    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < selected_ensemble_ids.size(); idx++) {
        int ens = selected_ensemble_ids[idx];
        // Use pre-filtered particle indices
        const auto& particle_indices = ensemble_particle_indices[ens];
        int part_num = particle_indices.size();

        if (part_num == 0) continue; // Skip empty ensembles

        // Create filename: output_ens/ens_XXX_timestep_XXXXX.vtk
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

        vtkFile << "# vtk DataFile Version 4.0\n";
        vtkFile << "Ensemble " << ens << " particle data\n";
        vtkFile << "BINARY\n";
        vtkFile << "DATASET POLYDATA\n";

        vtkFile << "POINTS " << part_num << " float\n";
        float zsum = 0.0;
        for (int idx : particle_indices){
            float x = -179.0 + part[idx].x*0.5;
            float y = -90.0 + part[idx].y*0.5;
            float z = part[idx].z/3000.0;
            zsum += part[idx].z;

            swapByteOrder(x);
            swapByteOrder(y);
            swapByteOrder(z);

            vtkFile.write(reinterpret_cast<char*>(&x), sizeof(float));
            vtkFile.write(reinterpret_cast<char*>(&y), sizeof(float));
            vtkFile.write(reinterpret_cast<char*>(&z), sizeof(float));
        }

        vtkFile << "POINT_DATA " << part_num << "\n";

        vtkFile << "SCALARS time_idx int 1\n";
        vtkFile << "LOOKUP_TABLE default\n";
        for (int i = 0; i < nop; ++i){
            if(!part[i].flag) continue;
            int vval = part[i].timeidx;
            swapByteOrder(vval);
            vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(int));
        }

        vtkFile << "SCALARS Q float 1\n";
        vtkFile << "LOOKUP_TABLE default\n";

        for (int idx : particle_indices){
            float vval = part[idx].conc;
            swapByteOrder(vval);
            vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
        }

        vtkFile << "SCALARS time_idx int 1\n";
        vtkFile << "LOOKUP_TABLE default\n";
        for (int idx : particle_indices){
            int vval = part[idx].timeidx;
            swapByteOrder(vval);
            vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(int));
        }

        vtkFile.close();
    } // end of ensemble loop
}

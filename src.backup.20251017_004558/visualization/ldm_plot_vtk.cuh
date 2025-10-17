/**
 * @file ldm_plot_vtk.cuh
 * @brief VTK output functions for particle visualization
 *
 * @details Provides VTK (Visualization Toolkit) file format output for
 *          particle data visualization in 3D. Supports both single-mode
 *          and ensemble-mode simulations with optimized parallel output.
 *
 * @note VTK output can be computationally expensive; use enable_vtk_output
 *       flag to control when output is generated
 *
 * @see https://vtk.org for VTK file format specification
 */

#pragma once

#ifndef LDM_CLASS_DECLARED
#error "Include the header that declares class LDM before including ldm_plot_vtk.cuh"
#endif

// Forward declarations for LDM class methods

/**
 * @method LDM::outputParticlesBinaryMPI
 * @brief Output particle data to VTK format for single-mode simulation
 *
 * @details Writes active particles to a VTK POLYDATA file in binary format.
 *          The function performs the following operations:
 *          1. Copies particle data from GPU to host memory
 *          2. Counts active particles (flag == 1)
 *          3. Creates output directory if needed
 *          4. Writes VTK header and particle positions
 *          5. Writes particle properties (concentration, time index)
 *
 * @param[in] timestep Current simulation timestep for file naming
 *
 * @pre Particles must be allocated and initialized on GPU (d_part)
 * @pre Output directory "output/plot_vtk_prior" will be created if missing
 *
 * @post VTK file created at output/plot_vtk_prior/plot_XXXXX.vtk
 *
 * @note Coordinates are converted from GFS grid units to geographic (lat/lon)
 * @note Z-coordinates are scaled by 1/3000 for better visualization
 * @note Binary data uses big-endian byte order (swapped on x86)
 *
 * @algorithm
 *   1. cudaMemcpy: GPU → Host particle transfer
 *   2. Count active particles (flag == 1)
 *   3. mkdir: Create output directory
 *   4. Write VTK ASCII header
 *   5. Write binary POINTS data (x, y, z)
 *   6. Write binary POINT_DATA (Q, time_idx)
 *
 * @performance
 *   - Memory transfer: O(N) where N = total particles
 *   - File I/O: ~0.5-2s for 1M particles
 *   - Disk usage: ~50-100MB per timestep for 1M particles
 */
void outputParticlesBinaryMPI(int timestep);

/**
 * @method LDM::outputParticlesBinaryMPI_ens
 * @brief Output ensemble particle data to VTK format with parallel I/O
 *
 * @details Writes multiple ensemble members to separate VTK files using
 *          OpenMP parallelization for improved performance. Only selected
 *          ensembles (stored in selected_ensemble_ids) are written to
 *          reduce I/O overhead.
 *
 * @param[in] timestep Current simulation timestep for file naming
 *
 * @pre Ensemble mode must be enabled (is_ensemble_mode == true)
 * @pre selected_ensemble_ids must be populated (typically 3 ensembles)
 * @pre Particles must have valid ensemble_id field
 *
 * @post VTK files created at output/plot_vtk_ens/ens_XXX_timestep_XXXXX.vtk
 *
 * @note Uses OpenMP with 50 threads for parallel file writing
 * @note Only writes particles with ensemble_id in selected_ensemble_ids
 * @note Pre-filters particles by ensemble to avoid redundant iterations
 *
 * @algorithm
 *   1. cudaMemcpy: GPU → Host (all particles)
 *   2. Pre-filter particles by ensemble_id
 *   3. #pragma omp parallel for: Parallel ensemble loop
 *   4. For each selected ensemble:
 *      - Create ensemble-specific VTK file
 *      - Write binary POINTS and POINT_DATA
 *   5. All files written concurrently
 *
 * @performance
 *   - Parallelization: 50 OpenMP threads
 *   - Speedup: ~10-20x vs sequential for 100 ensembles
 *   - Memory: All particles loaded in host memory
 *   - Disk usage: ~50-100MB per ensemble per timestep
 *
 * @warning Pre-filtering trades memory for speed (creates index vectors)
 * @warning Concurrent file I/O may saturate disk bandwidth
 */
void outputParticlesBinaryMPI_ens(int timestep);


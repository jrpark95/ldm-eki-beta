/**
 * @file ldm_plot_vtk.cuh
 * @brief VTK output functions for particle visualization
 * @author Juryong Park
 * @date 2025
 *
 * @details Provides VTK (Visualization Toolkit) file format output for
 *          particle data visualization in 3D. Supports both single-mode
 *          and ensemble-mode simulations with optimized parallel output.
 *
 *          This module generates VTK POLYDATA files in Legacy format (VTK 4.0)
 *          with binary data encoding. The output can be visualized using
 *          ParaView, VisIt, or other VTK-compatible tools.
 *
 * @section vtk_format VTK File Format Specification
 *
 * **Format Type**: VTK Legacy Format (Version 4.0)
 * **Dataset Type**: POLYDATA (unstructured point cloud)
 * **Encoding**: Binary (big-endian byte order)
 *
 * **File Structure**:
 * 1. **Header** (ASCII text):
 *    - Version: "# vtk DataFile Version 4.0"
 *    - Description: "particle data" or "Ensemble N particle data"
 *    - Format: "BINARY"
 *    - Dataset: "DATASET POLYDATA"
 *
 * 2. **Geometry Section** (Binary):
 *    - POINTS: Particle positions (x, y, z) as float triplets
 *    - Count: Variable (active particles only)
 *
 * 3. **Attributes Section** (Binary):
 *    - POINT_DATA: Per-particle scalar fields
 *    - Q (float): Particle concentration [Bq/m³]
 *    - time_idx (int): Emission time index
 *
 * @section coord_system Coordinate Systems
 *
 * **GFS Grid Coordinates** (Internal representation):
 * - x: [0, 719] grid units (0.5° resolution)
 * - y: [0, 359] grid units (0.5° resolution)
 * - z: [0, ~10000] meters above ground level
 *
 * **Geographic Coordinates** (VTK output):
 * - x: [-179°, +180°] longitude
 * - y: [-90°, +90°] latitude
 * - z: [0, ~3.3] scaled altitude (z/3000 for better visualization)
 *
 * **Conversion Formula**:
 * - lon = -179.0 + grid_x * 0.5
 * - lat = -90.0 + grid_y * 0.5
 * - alt_scaled = grid_z / 3000.0
 *
 * @section output_modes Output Modes
 *
 * **Single Mode** (outputParticlesBinaryMPI):
 * - Output directory: output/plot_vtk_prior/
 * - Filename pattern: plot_XXXXX.vtk (XXXXX = timestep)
 * - Use case: Initial "truth" simulation, validation runs
 *
 * **Ensemble Mode** (outputParticlesBinaryMPI_ens):
 * - Output directory: output/plot_vtk_ens/
 * - Filename pattern: ens_XXX_timestep_XXXXX.vtk
 * - Parallelization: 50 OpenMP threads
 * - Selected ensembles only (typically 3 out of 100)
 *
 * @note VTK output can be computationally expensive; use enable_vtk_output
 *       flag to control when output is generated (disabled during intermediate
 *       EKI iterations, enabled for final iteration)
 *
 * @performance
 * - File I/O: ~0.5-2s per timestep for 1M particles
 * - Disk usage: ~50-100MB per timestep per ensemble
 * - Parallel speedup: ~10-20x for ensemble mode
 *
 * @see https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
 */

#pragma once

#ifndef LDM_CLASS_DECLARED
#error "Include the header that declares class LDM before including ldm_plot_vtk.cuh"
#endif

// ============================================================================
// VTK Output Functions
// ============================================================================

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


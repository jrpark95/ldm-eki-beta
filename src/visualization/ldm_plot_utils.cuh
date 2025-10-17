/**
 * @file ldm_plot_utils.cuh
 * @brief Utility functions for visualization and validation output
 * @author Juryong Park
 * @date 2025
 *
 * @details Provides two categories of functions:
 *
 *          **1. VTK Utilities** (Production code)
 *          - Byte-order conversion for VTK binary format
 *          - Active particle counting for file headers
 *          - Used in all VTK output operations
 *
 *          **2. Validation Functions** (Development/debugging)
 *          - Time-series logging of particle concentrations
 *          - Nuclide decay analysis and CRAM verification
 *          - Gridded concentration export for model comparison
 *          - CSV output for external analysis
 *
 * @note Validation functions can be disabled in production builds by
 *       commenting out their calls in the main simulation loop
 *
 * @performance Validation functions involve GPU→Host copies and file I/O,
 *              potentially slowing down simulation by 5-10%
 */

#pragma once

#ifndef LDM_CLASS_DECLARED
#error "Include the header that declares class LDM before including ldm_plot_utils.cuh"
#endif

// ============================================================================
// VTK Utility Functions
// ============================================================================

/**
 * @method LDM::countActiveParticles
 * @brief Count number of active particles in simulation
 *
 * @details Iterates through host particle array and counts particles with
 *          flag == 1 (active). Used to determine the POINTS count in VTK
 *          file headers before writing geometry data.
 *
 * @return int Number of active particles (flag == 1)
 *
 * @pre Particles must be copied from GPU to host (part vector)
 * @complexity O(N) where N = total particle count (nop)
 * @note Operates on host memory only (part vector, not d_part)
 *
 * @example
 * cudaMemcpy(part.data(), d_part, nop * sizeof(LDMpart), cudaMemcpyDeviceToHost);
 * int active_count = countActiveParticles();
 * // Use active_count in VTK header: "POINTS <active_count> float"
 */
int countActiveParticles();

/**
 * @method LDM::swapByteOrder (float overload)
 * @brief Convert float from little-endian to big-endian for VTK binary format
 *
 * @details VTK Legacy format specifies big-endian byte order for binary data.
 *          On x86 systems (little-endian), we must reverse the byte order
 *          before writing to file. This function swaps 4 bytes in-place:
 *          [B0 B1 B2 B3] → [B3 B2 B1 B0]
 *
 * @param[in,out] value Float value to byte-swap (modified in-place)
 *
 * @complexity O(1) - two byte swaps
 * @note Function is a no-op on big-endian systems (but called anyway for portability)
 * @note Must be called before every binary write to VTK file
 *
 * @example
 * float x = 123.456f;
 * swapByteOrder(x);
 * vtkFile.write(reinterpret_cast<char*>(&x), sizeof(float));
 */
void swapByteOrder(float& value);

/**
 * @method LDM::swapByteOrder (int overload)
 * @brief Convert integer from little-endian to big-endian for VTK binary format
 *
 * @details Integer overload of swapByteOrder. Used for integer attributes
 *          like time_idx in VTK POINT_DATA sections.
 *
 * @param[in,out] value Integer value to byte-swap (modified in-place)
 *
 * @complexity O(1) - two byte swaps
 * @note Same byte-swapping logic as float version
 */
void swapByteOrder(int& value);

// ============================================================================
// Validation and Logging Functions
// ============================================================================

/**
 * @method LDM::log_first_particle_concentrations
 * @brief Log first active particle's nuclide concentrations over time
 *
 * @details Writes time-series data of first active particle's concentrations
 *          to CSV file for validation of nuclide transport and decay.
 *
 * @param[in] timestep Current simulation timestep
 * @param[in] currentTime Current simulation time [seconds]
 *
 * @post Creates/appends to validation/first_particle_concentrations.csv
 *
 * @note Only logs first active particle to reduce file size
 * @note CSV format: timestep,time(s),total_conc,nuclide1,nuclide2,...
 */
void log_first_particle_concentrations(int timestep, float currentTime);

/**
 * @method LDM::log_all_particles_nuclide_ratios
 * @brief Log aggregate nuclide ratios for all active particles
 *
 * @details Computes total concentrations and ratios across all active particles,
 *          useful for mass conservation checks and decay validation.
 *
 * @param[in] timestep Current simulation timestep
 * @param[in] currentTime Current simulation time [seconds]
 *
 * @post Creates/appends to validation/all_particles_nuclide_ratios.csv
 *
 * @invariants Total mass should decrease monotonically due to decay
 * @note CSV includes both total concentrations and ratios
 */
void log_all_particles_nuclide_ratios(int timestep, float currentTime);

/**
 * @method LDM::log_first_particle_cram_detail
 * @brief Log detailed CRAM decay calculation for first particle
 *
 * @details Records pre/post decay concentrations with half-lives and decay factors
 *          to validate CRAM (Chebyshev Rational Approximation Method) accuracy.
 *
 * @param[in] timestep Current simulation timestep
 * @param[in] currentTime Current simulation time [seconds]
 * @param[in] dt_used Time step used in CRAM calculation [seconds]
 *
 * @post Creates/appends to validation/first_particle_cram_detail.csv
 *
 * @note Includes mass conservation ratio check
 * @see ldm_cram2.cuh for CRAM implementation details
 */
void log_first_particle_cram_detail(int timestep, float currentTime, float dt_used);

/**
 * @method LDM::log_first_particle_decay_analysis
 * @brief Compare observed vs theoretical decay for each nuclide
 *
 * @details Calculates theoretical concentration based on exponential decay
 *          formula and compares with CRAM results to quantify accuracy.
 *
 * @param[in] timestep Current simulation timestep
 * @param[in] currentTime Current simulation time [seconds]
 *
 * @post Creates/appends to validation/first_particle_decay_analysis.csv
 *
 * @algorithm
 *   For each nuclide:
 *   1. theoretical = C0 * exp(-λ * age)
 *   2. relative_error = (observed - theoretical) / theoretical * 100%
 *
 * @note Useful for verifying CRAM accuracy against simple decay
 */
void log_first_particle_decay_analysis(int timestep, float currentTime);

// ============================================================================
// Validation Data Export Functions
// ============================================================================

/**
 * @method LDM::exportValidationData
 * @brief Master function for exporting validation datasets
 *
 * @details Coordinates export of gridded concentration data and nuclide totals
 *          for CRAM validation against reference implementations.
 *
 * @param[in] timestep Current simulation timestep
 * @param[in] currentTime Current simulation time [seconds]
 *
 * @post Calls exportConcentrationGrid() and exportNuclideTotal()
 *
 * @note Grid export triggered at selected timesteps to save disk space
 * @note Prints progress message every 100 timesteps
 */
void exportValidationData(int timestep, float currentTime);

/**
 * @method LDM::exportConcentrationGrid
 * @brief Export 3D gridded concentration field for spatial analysis
 *
 * @details Maps particle cloud to regular 3D grid for easier comparison with
 *          other models and observational data. Grid domain is designed for
 *          Fukushima region but can be adjusted in implementation.
 *
 * @param[in] timestep Current simulation timestep
 * @param[in] currentTime Current simulation time [seconds]
 *
 * @post Creates validation/concentration_grid_XXXXX.csv
 *
 * @grid_specification
 * - Domain: 139-143°E, 36-39°N, 0-2000m
 * - Dimensions: 100 × 100 × 20 cells
 * - Resolution: 0.04° × 0.03° × 100m
 * - Total cells: 200,000
 *
 * @output_format CSV with columns:
 * - x_index, y_index, z_index: Grid cell indices
 * - lon, lat, alt: Cell center coordinates
 * - concentration: Accumulated particle concentration [Bq/m³]
 * - particle_count: Number of particles in cell
 *
 * @optimization Sparse output: Only cells with concentration > 0 are written
 *
 * @algorithm
 *   1. Initialize 100×100×20 grid with zeros
 *   2. For each active particle:
 *      - Convert (x, y, z) to grid indices (ix, iy, iz)
 *      - Accumulate concentration in grid[ix][iy][iz]
 *      - Increment particle count
 *   3. Write non-zero cells to CSV
 *
 * @performance O(N) where N = active particles
 */
void exportConcentrationGrid(int timestep, float currentTime);

/**
 * @method LDM::exportNuclideTotal
 * @brief Export total concentration for each nuclide
 *
 * @details Sums concentrations across all active particles for each nuclide,
 *          providing time-series data for mass conservation validation.
 *
 * @param[in] timestep Current simulation timestep
 * @param[in] currentTime Current simulation time [seconds]
 *
 * @post Creates/appends to validation/nuclide_totals.csv
 *
 * @note CSV format: timestep,time,active_particles,total_conc,total_nuc1,...
 * @note Useful for checking if total mass is conserved (except decay)
 *
 * @invariants Sum should decrease monotonically for decaying nuclides
 */
void exportNuclideTotal(int timestep, float currentTime);

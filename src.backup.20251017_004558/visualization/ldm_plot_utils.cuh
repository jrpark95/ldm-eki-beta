/**
 * @file ldm_plot_utils.cuh
 * @brief Utility functions for visualization and validation output
 *
 * @details Provides helper functions for VTK output (byte swapping, particle counting)
 *          and validation/logging functions for debugging particle concentrations
 *          and nuclide decay processes.
 *
 * @note Validation functions are used for development/debugging and can be
 *       disabled in production builds for better performance
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
 * @details Iterates through particle array and counts particles with flag == 1.
 *          Used before VTK output to determine file size and allocation.
 *
 * @return int Number of active particles
 *
 * @complexity O(N) where N = total particle count
 * @note Operates on host memory (part vector)
 */
int countActiveParticles();

/**
 * @method LDM::swapByteOrder (float overload)
 * @brief Convert float from little-endian to big-endian for VTK binary format
 *
 * @details VTK binary format requires big-endian byte order. This function
 *          swaps bytes in-place for x86 systems (little-endian).
 *
 * @param[in,out] value Float value to byte-swap
 *
 * @note Modifies value in-place
 * @complexity O(1)
 */
void swapByteOrder(float& value);

/**
 * @method LDM::swapByteOrder (int overload)
 * @brief Convert integer from little-endian to big-endian for VTK binary format
 *
 * @param[in,out] value Integer value to byte-swap
 *
 * @note Modifies value in-place
 * @complexity O(1)
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
 * @brief Export 3D gridded concentration field
 *
 * @details Maps particles to 100×100×20 grid covering Fukushima region
 *          (139-143°E, 36-39°N, 0-2000m) and exports to CSV.
 *
 * @param[in] timestep Current simulation timestep
 * @param[in] currentTime Current simulation time [seconds]
 *
 * @post Creates validation/concentration_grid_XXXXX.csv
 *
 * @note Grid resolution: 0.04° × 0.03° × 100m
 * @note Only exports cells with concentration > 0 (sparse format)
 *
 * @algorithm
 *   1. Initialize 100×100×20 grid
 *   2. For each active particle:
 *      - Convert position to grid indices
 *      - Accumulate concentration in cell
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

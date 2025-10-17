/******************************************************************************
 * @file ldm_func_output.cuh
 * @brief Output handling, observation system, and logging for EKI data assimilation
 *
 * This module implements the observation collection system for Ensemble Kalman
 * Inversion (EKI), which is central to the data assimilation framework. Key
 * components include:
 *
 * Receptor Observation System:
 * - Defines virtual receptor locations (lat/lon coordinates) where observations are collected
 * - Accumulates particle concentrations within capture radius of each receptor
 * - Computes gamma dose rates from multi-nuclide particle contributions
 * - Supports both single-mode (true simulation) and ensemble-mode (parallel simulations)
 *
 * Time Integration:
 * - Observations accumulated over specified time intervals (e.g., 15 minutes)
 * - Temporal aggregation matches observation data time resolution
 * - Each timestep contributes to cumulative dose over observation period
 *
 * Grid Receptor Mode (Optional):
 * - Regular grid of receptors for spatial field reconstruction
 * - Used for debugging and validation of observation system
 * - Generates CSV time series for each grid point
 *
 * Data Flow for EKI:
 * 1. Initialize observation system with receptor locations and parameters
 * 2. During simulation: Accumulate contributions at each timestep
 * 3. At observation intervals: Transfer GPU results to host memory
 * 4. Write observations to shared memory for Python EKI process
 * 5. Reset accumulators for next observation period
 *
 * @note This header only provides forward declarations.
 *       Actual method declarations are in ldm.cuh
 *       Implementations are in ldm_func_output.cu
 *
 * @see ldm.cuh for LDM class method declarations
 * @see ldm_func_output.cu for implementation details
 * @see ldm_eki_writer.cu for shared memory IPC mechanism
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/

#pragma once

// Forward declaration only - method declarations are in ldm.cuh
class LDM;

/******************************************************************************
 * @brief Test function for verifying cross-compilation-unit global access
 *
 * Diagnostic function used during development to ensure the global log file
 * pointer (g_log_file) is correctly accessible from this compilation unit.
 * Should only be called during initialization or debugging phases.
 *
 * @post Writes test message to log file if accessible
 *
 * @note For development/testing only - not used in production runs
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void test_g_logonly_from_output_module();

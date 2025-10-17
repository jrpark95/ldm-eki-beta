/******************************************************************************
 * @file ldm_func_simulation.cuh
 * @brief Main simulation loop and execution control for LDM-EKI framework
 *
 * This module provides the core simulation execution functions for the
 * Lagrangian Dispersion Model (LDM) coupled with Ensemble Kalman Inversion (EKI).
 * It manages:
 *
 * - Main time-stepping loop for particle advection
 * - Meteorological data updates during simulation
 * - Particle activation scheduling based on emission time series
 * - Integration with EKI observation system for data assimilation
 * - VTK output control for visualization
 *
 * The simulation engine supports three operational modes:
 * 1. Standard mode: Regular forward simulation with on-demand met data loading
 * 2. EKI single mode: Initial "true" simulation for generating reference observations
 * 3. EKI ensemble mode: Parallel simulation of multiple ensemble members for inversion
 *
 * Key Features:
 * - GPU-accelerated particle advection using CUDA kernels
 * - Preloaded meteorological data in EKI mode for fast ensemble iterations
 * - Flexible observation collection at receptor locations
 * - Progress reporting and performance monitoring
 *
 * @note This header only provides forward declarations.
 *       Actual method declarations are in ldm.cuh
 *       Implementations are in ldm_func_simulation.cu
 *
 * @see ldm.cuh for LDM class method declarations
 * @see ldm_func_simulation.cu for implementation details
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/

#pragma once

// Forward declaration only - method declarations are in ldm.cuh
class LDM;

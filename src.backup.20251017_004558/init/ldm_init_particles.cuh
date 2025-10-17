/**
 * @file ldm_init_particles.cuh
 * @brief Particle initialization functions for LDM simulation
 *
 * @details Provides particle initialization for different simulation modes:
 *          - Standard mode: concentration-based particle distribution
 *          - EKI single mode: true emission time series
 *          - EKI ensemble mode: multiple ensemble member initialization
 *
 * @note All initialization functions properly set up multi-nuclide concentrations
 */

#pragma once

#ifndef LDM_CLASS_DECLARED
#error "Include the header that declares class LDM before including ldm_init_particles.cuh"
#endif

#include <chrono>

/**
 * @method LDM::initializeParticles
 * @brief Initialize particles for standard simulation mode
 *
 * @details Creates particles based on source locations and concentration values
 *          defined in source.txt. Particles are distributed evenly across
 *          release cases with randomized properties.
 *
 * @pre concentrations vector must be loaded from source.txt
 * @pre sources vector must be loaded from source.txt
 * @pre nop (number of particles) must be set in configuration
 *
 * @post part vector populated with initialized particles
 * @post Particles sorted by timeidx for proper activation sequencing
 *
 * @algorithm
 *   1. Calculate particles per concentration case
 *   2. For each case:
 *      - Get source location and emission value
 *      - Generate particles with random sizes
 *      - Initialize multi-nuclide concentrations
 *   3. Sort particles by timeidx
 *
 * @note Particle sizes sampled from normal distribution
 * @note Multi-nuclide concentrations set using initial ratios from config
 */
void initializeParticles();

/**
 * @method LDM::initializeParticlesEKI
 * @brief Initialize particles for EKI single-mode simulation
 *
 * @details Creates particles using true_emissions time series from EKI config.
 *          Particles are distributed across time intervals to represent
 *          time-varying emission rates.
 *
 * @pre g_eki.true_emissions must be loaded from eki_settings.txt
 * @pre sources vector must contain at least one source location
 * @pre nop must be divisible by number of time intervals
 *
 * @post part vector cleared and repopulated
 * @post Particles sorted by timeidx
 * @post All particles set to timeidx for immediate activation
 *
 * @algorithm
 *   1. Calculate particles per time interval
 *   2. For each time interval:
 *      - Get emission rate from true_emissions
 *      - Generate particles with that emission value
 *      - Set multi-nuclide concentrations
 *   3. Sort by timeidx
 *
 * @note Different from ensemble mode: all particles activate immediately
 * @note Used for generating "truth" observations in EKI workflow
 */
void initializeParticlesEKI();

/**
 * @method LDM::initializeParticlesEKI_AllEnsembles
 * @brief Initialize particles for EKI ensemble simulation mode
 *
 * @details Creates particles for multiple ensemble members simultaneously.
 *          Each ensemble has its own emission time series, and particles
 *          are tagged with ensemble_id for independent simulation.
 *
 * @param[in] ensemble_states Emission rates [num_ensembles × num_timesteps]
 * @param[in] num_ensembles Number of ensemble members (typically 50-100)
 * @param[in] num_timesteps Number of time intervals in emission series
 *
 * @pre ensemble_states must be row-major: states[ens*num_timesteps + t]
 * @pre sources vector must contain at least one source location
 * @pre Memory: total_particles = nop * num_ensembles can be large (millions)
 *
 * @post part vector contains num_ensembles * nop particles
 * @post Particles sorted by ensemble_id, then timeidx
 * @post Each particle has valid ensemble_id field (0 to num_ensembles-1)
 *
 * @algorithm
 *   1. Reserve memory for all particles
 *   2. For each ensemble:
 *      For each time interval:
 *        - Get emission rate for this ensemble/time
 *        - Generate particles_per_timestep particles
 *        - Set ensemble_id and timeidx
 *        - Initialize multi-nuclide concentrations
 *   3. Sort by ensemble_id (primary) and timeidx (secondary)
 *
 * @performance
 *   - Memory: O(num_ensembles * nop) particles
 *   - Time: O(num_ensembles * nop) particle generation
 *   - Sorting: O(N log N) where N = total particles
 *
 * @note Critical: Particles MUST be sorted by ensemble_id for GPU kernel efficiency
 * @note Each ensemble simulates independently with its own emission series
 * @note timeidx ranges from 1 to particles_per_ensemble for each ensemble
 *
 * @warning Large ensemble sizes (>100) with large nop (>100k) can exceed memory
 */
void initializeParticlesEKI_AllEnsembles(float* ensemble_states, int num_ensembles, int num_timesteps);

/**
 * @method LDM::calculateSettlingVelocity
 * @brief Calculate particle settling velocity using size distribution
 *
 * @details Computes average settling velocity (vsetaver) and Cunningham factor
 *          for particles with given size distribution using Stokes law and
 *          slip correction.
 *
 * @pre g_mpi particle properties must be loaded from configuration
 * @pre Particle radius (radi) and density (prho) must be set
 *
 * @post vsetaver: average settling velocity [m/s]
 * @post cunningham: average Cunningham correction factor
 * @post Both values copied to device constant memory
 *
 * @algorithm
 *   1. Generate size fractions using log-normal distribution
 *   2. For each size fraction:
 *      - Calculate Knudsen number
 *      - Compute Cunningham correction factor
 *      - Calculate settling velocity using Stokes law
 *   3. Average over all size fractions
 *   4. Upload to GPU constant memory
 *
 * @note Uses NI=11 size intervals for integration
 * @note Special handling for radi=0 (gas-phase species)
 * @note Cunningham factor accounts for slip flow in small particles
 *
 * @see calculateAverageSettlingVelocity() for legacy version
 */
void calculateSettlingVelocity();

/**
 * @method LDM::calculateAverageSettlingVelocity
 * @brief Legacy function for average settling velocity calculation
 *
 * @details Similar to calculateSettlingVelocity() but with hardcoded
 *          particle properties. Kept for backward compatibility.
 *
 * @deprecated Use calculateSettlingVelocity() instead for configurable properties
 *
 * @post vsetaver and cunningham set to calculated values
 * @post Values uploaded to device constant memory
 */
void calculateAverageSettlingVelocity();

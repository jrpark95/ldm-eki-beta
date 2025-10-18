/**
 * @file ldm_kernels_device.cuh
 * @brief Device utility functions for LDM particle simulation
 *
 * @details This module contains device-side helper functions used across
 *          all kernel modules. Includes nuclear decay, meteorological
 *          calculations, random number generation, and viscosity models.
 */

#pragma once

#include <curand_kernel.h>
#include "../../core/ldm.cuh"
#include "../cram/ldm_kernels_cram.cuh"
#include "../../physics/ldm_nuclides.cuh"

// ============================================================================
// NUCLEAR DECAY FUNCTIONS
// ============================================================================

/**
 * @device nuclear_decay_optimized_inline
 * @brief Simplified 2-nuclide decay for testing (Sr-92 → Y-92)
 *
 * @details TEST FUNCTION - NOT USED IN PRODUCTION
 *          Implements exponential decay for Sr-92 → Y-92 chain.
 *          Production code uses nuclear_decay_optimized() from ldm_cram.cuh
 *
 * @param[in,out] exp_matrix     CRAM exponential matrix (unused in this version)
 * @param[in,out] concentration  Nuclide concentrations [MAX_NUCLIDES]
 * @param[in]     no_clamp       If true, skip clamping checks
 *
 * @note Uses d_dt global constant for time step
 * @warning Handles numerical errors by clamping to small positive values
 */
__device__ void nuclear_decay_optimized_inline(float* exp_matrix, float* concentration, bool no_clamp = false);

// ============================================================================
// METEOROLOGICAL UTILITY FUNCTIONS
// ============================================================================

/**
 * @device k_f_esi
 * @brief Calculate saturation vapor pressure over ice
 *
 * @details Uses enhanced Teten's formula for ice surfaces.
 *          Applies enhancement factor and exponential temperature relationship.
 *
 * @param[in] p  Pressure [Pa]
 * @param[in] t  Temperature [K]
 *
 * @return Saturation vapor pressure over ice [Pa]
 *
 * @complexity O(1)
 */
__device__ float k_f_esi(float p, float t);

/**
 * @device k_f_esl
 * @brief Calculate saturation vapor pressure over liquid water
 *
 * @details Uses enhanced Teten's formula for liquid surfaces.
 *          Applies enhancement factor and exponential temperature relationship.
 *
 * @param[in] p  Pressure [Pa]
 * @param[in] t  Temperature [K]
 *
 * @return Saturation vapor pressure over liquid water [Pa]
 *
 * @complexity O(1)
 */
__device__ float k_f_esl(float p, float t);

/**
 * @device k_f_qvsat
 * @brief Calculate saturation mixing ratio
 *
 * @details Computes saturation water vapor mixing ratio.
 *          Switches between liquid and ice formulas at 253.15K
 *          to account for supercooled water.
 *
 * @param[in] p  Pressure [Pa]
 * @param[in] t  Temperature [K]
 *
 * @return Saturation mixing ratio [dimensionless]
 *
 * @note Uses rd/rv ratio where rd=287 m²/(s²*K), rv=461 m²/(s²*K)
 * @warning Returns 1.0 if denominator is zero to avoid division error
 */
__device__ float k_f_qvsat(float p, float t);

// ============================================================================
// RANDOM NUMBER GENERATION
// ============================================================================

/**
 * @device getRand
 * @brief Generate uniform random number in [0,1]
 *
 * @param[in,out] states  CUDA random number generator state
 *
 * @return Uniform random float in [0,1]
 *
 * @complexity O(1)
 */
__device__ __forceinline__ float getRand(curandState* states) {
    return curand_uniform(states);
}

/**
 * @device GaussianRand
 * @brief Generate Gaussian random number using Box-Muller transform
 *
 * @details Uses two uniform random numbers to generate one
 *          Gaussian-distributed random number via Box-Muller method.
 *
 * @param[in,out] states  CUDA random number generator state
 * @param[in]     mu      Mean of Gaussian distribution
 * @param[in]     stdv    Standard deviation of Gaussian distribution
 *
 * @return Gaussian random float with specified mean and stddev
 *
 * @complexity O(1)
 * @note Calls getRand() twice internally
 */
__device__ __forceinline__ float GaussianRand(curandState* states, float mu, float stdv) {
    float u1 = getRand(states);
    float u2 = getRand(states);
    float mag = stdv*sqrt(-2.0*log(u1));
    return mag*cos(2*PI*u2)+mu;
}

// ============================================================================
// INITIALIZATION KERNELS
// ============================================================================

/**
 * @kernel init_curand_states
 * @brief Initialize random number generator states for all particles
 *
 * @details Sets up cuRAND state for each particle using time-dependent seed.
 *          Seed combines simulation time and particle index for uniqueness.
 *
 * @param[in,out] d_part         Device array of particles
 * @param[in]     t0             Initial time for seed generation [seconds]
 * @param[in]     num_particles  Total number of particles
 *
 * @grid_config
 *   - Block size: 256 threads
 *   - Grid size: (num_particles + 255) / 256
 */
__global__ void initCurandStates(LDM::LDMpart* d_part, float t0, int num_particles);

/**
 * @kernel update_particle_flags
 * @brief Activate particles progressively based on activation ratio
 *
 * @details Single-mode particle activation. Sets particle.flag = 1
 *          for particles within the active index range.
 *
 * @param[in,out] d_part          Device array of particles
 * @param[in]     activationRatio Fraction of particles to activate [0,1]
 * @param[in]     num_particles   Total number of particles
 *
 * @grid_config
 *   - Block size: 256 threads
 *   - Grid size: (num_particles + 255) / 256
 */
__global__ void updateParticleFlags(LDM::LDMpart* d_part, float activationRatio, int num_particles);

/**
 * @kernel update_particle_flags_ens
 * @brief Activate particles progressively in ensemble mode
 *
 * @details Ensemble-mode particle activation. Activates particles
 *          within each ensemble member based on local time index.
 *
 * @param[in,out] d_part                  Device array of particles
 * @param[in]     activationRatio         Fraction to activate [0,1]
 * @param[in]     total_particles         Total particle count across ensembles
 * @param[in]     particles_per_ensemble  Particles in each ensemble member
 *
 * @grid_config
 *   - Block size: 256 threads
 *   - Grid size: (total_particles + 255) / 256
 *
 * @note Activates based on particle.timeidx within each ensemble
 * @see updateParticleFlags() for single-mode version
 */
__global__ void updateParticleFlagsEnsemble(
    LDM::LDMpart* d_part, float activationRatio,
    int total_particles, int particles_per_ensemble);

// ============================================================================
// PHYSICAL PROPERTY FUNCTIONS
// ============================================================================

/**
 * @device Dynamic_viscosity
 * @brief Calculate dynamic viscosity of air using Sutherland's formula
 *
 * @details Computes air viscosity as function of temperature.
 *          Uses Sutherland's law with reference values:
 *          - Reference temperature: 291.15 K
 *          - Sutherland constant: 120.0 K
 *          - Reference viscosity: 1.827e-5 Pa·s
 *
 * @param[in] temp  Air temperature [K]
 *
 * @return Dynamic viscosity [Pa·s]
 *
 * @complexity O(1)
 * @note Uses pow() for 1.5 exponent
 */
__device__ __forceinline__ float Dynamic_viscosity(float temp) {
    float c = 120.0;
    float t0 = 291.15;
    float eta0 = 1.827e-5;
    return eta0 * ((t0+c)/(temp+c)) * pow((temp/t0), 1.5);
}

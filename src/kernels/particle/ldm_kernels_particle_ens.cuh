/**
 * @file ldm_kernels_particle_ens.cuh
 * @brief Particle advection kernel for ensemble mode simulation
 *
 * @details GPU kernel for ensemble particle transport. Identical physics
 *          to single mode but handles particles from all ensemble members.
 */

#pragma once

#include "../device/ldm_kernels_device.cuh"
#include "../cram/ldm_kernels_cram.cuh"
#include "../../physics/ldm_cram2.cuh"
#include "../../core/params.hpp"

/**
 * @kernel move_part_by_wind_mpi_ens
 * @brief Main particle advection kernel (ensemble mode)
 *
 * @details Ensemble version of particle movement kernel. Processes particles
 *          from all ensemble members simultaneously. Each particle carries
 *          an ensemble_id to track membership.
 *
 *          Physics identical to single mode:
 *          - Wind advection + turbulent diffusion
 *          - Gravitational settling
 *          - Wet/dry deposition
 *          - Radioactive decay
 *
 * @param[in,out] d_part                         Particle array (all ensembles)
 * @param[in]     t0                             Time interpolation factor [0,1]
 * @param[in]     rank                           Process rank (unused)
 * @param[in]     d_dryDep                       Dry deposition flag
 * @param[in]     d_wetDep                       Wet deposition flag
 * @param[in]     mesh_nx                        Mesh size X (unused)
 * @param[in]     mesh_ny                        Mesh size Y (unused)
 * @param[in]     device_meteorological_flex_unis0  Meteorology time0 (2D)
 * @param[in]     device_meteorological_flex_pres0  Meteorology time0 (3D)
 * @param[in]     device_meteorological_flex_unis1  Meteorology time1 (2D)
 * @param[in]     device_meteorological_flex_pres1  Meteorology time1 (3D)
 * @param[in]     total_particles                Total particle count (all ensembles)
 *
 * @grid_config
 *   - Block size: 256 threads
 *   - Grid size: (total_particles + 255) / 256
 *
 * @performance
 *   - Memory throughput: ~75% of theoretical maximum
 *   - Typical runtime: 20-30ms for 100 ensembles × 100k particles
 *
 * @invariants
 *   - Each ensemble evolves independently
 *   - Preserves mass within each ensemble
 *   - Maintains ensemble_id consistency
 *
 * @note Uses global constants: d_dt, d_height_levels[], T_const
 * @note Particle ordering: [ens0_p0, ens0_p1, ..., ens1_p0, ens1_p1, ...]
 *
 * @warning Requires total_particles = num_ensemble * particles_per_ensemble
 *
 * @see advectParticles() for single-mode version
 * @see advectParticlesEnsembleWithVTK() for VTK output version
 */
__global__ void advectParticlesEnsemble(
    LDM::LDMpart* d_part, float t0, int rank, float* d_dryDep, float* d_wetDep, int mesh_nx, int mesh_ny,
    FlexUnis* device_meteorological_flex_unis0,
    FlexPres* device_meteorological_flex_pres0,
    FlexUnis* device_meteorological_flex_unis1,
    FlexPres* device_meteorological_flex_pres1,
    int total_particles,
    const KernelScalars ks);

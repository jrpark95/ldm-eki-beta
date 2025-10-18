/**
 * @file ldm_kernels_particle.cuh
 * @brief Particle advection kernel for single-mode simulation
 *
 * @details GPU kernel for particle transport including:
 *          - Wind advection (mean flow)
 *          - Turbulent diffusion (PBL parameterization)
 *          - Gravitational settling
 *          - Wet/dry deposition
 *          - Radioactive decay (CRAM method)
 */

#pragma once

#include "../device/ldm_kernels_device.cuh"
#include "../cram/ldm_kernels_cram.cuh"
#include "../../physics/ldm_cram2.cuh"
#include "../../core/params.hpp"

/**
 * @kernel move_part_by_wind_mpi
 * @brief Main particle advection kernel (single mode)
 *
 * @details Processes particle movement in parallel on GPU. Each thread handles
 *          one particle, computing:
 *          1. Meteorological interpolation (trilinear in space, linear in time)
 *          2. PBL turbulence parameterization (unstable/stable/neutral)
 *          3. Gravitational settling (Stokes drag with Re correction)
 *          4. Wet scavenging (in-cloud + below-cloud)
 *          5. Dry deposition (surface interaction)
 *          6. Radioactive decay (CRAM matrix exponential)
 *
 * @param[in,out] d_part                         Particle array
 * @param[in]     t0                             Time interpolation factor [0,1]
 * @param[in]     rank                           Process rank (unused)
 * @param[in]     d_dryDep                       Dry deposition flag
 * @param[in]     d_wetDep                       Wet deposition flag
 * @param[in]     mesh_nx                        Mesh size X (unused)
 * @param[in]     mesh_ny                        Mesh size Y (unused)
 * @param[in]     device_meteorological_flex_unis0  Meteorology time0 (2D fields)
 * @param[in]     device_meteorological_flex_pres0  Meteorology time0 (3D fields)
 * @param[in]     device_meteorological_flex_unis1  Meteorology time1 (2D fields)
 * @param[in]     device_meteorological_flex_pres1  Meteorology time1 (3D fields)
 *
 * @grid_config
 *   - Block size: 256 threads (optimal for SM 6.1+)
 *   - Grid size: (d_nop + 255) / 256
 *
 * @performance
 *   - Memory throughput: ~75% of theoretical maximum
 *   - Typical runtime: 2-3ms for 1M particles
 *   - Dominant cost: Meteorological interpolation
 *
 * @invariants
 *   - Preserves total mass (within numerical precision)
 *   - Maintains particle ordering
 *   - Handles PBL reflection (top/bottom)
 *
 * @note Uses global constants: d_nop, d_dt, d_height_levels[], T_const
 * @note Coordinate system: Grid indices → Lat/Lon conversion
 * @note PBL scheme: Hanna (1982) for unstable, stable, neutral
 *
 * @warning Skips inactive particles (flag == 0)
 * @warning Particles outside domain are flagged as inactive
 *
 * @see advectParticlesEnsemble() for ensemble version
 * @see advectParticlesWithVTK() for VTK output version
 */
__global__ void advectParticles(
    LDM::LDMpart* d_part, float t0, int rank, float* d_dryDep, float* d_wetDep, int mesh_nx, int mesh_ny,
    FlexUnis* device_meteorological_flex_unis0,
    FlexPres* device_meteorological_flex_pres0,
    FlexUnis* device_meteorological_flex_unis1,
    FlexPres* device_meteorological_flex_pres1,
    const KernelScalars ks);

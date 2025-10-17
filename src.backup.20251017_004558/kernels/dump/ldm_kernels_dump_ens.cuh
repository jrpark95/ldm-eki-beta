/**
 * @file ldm_kernels_dump_ens.cuh
 * @brief Particle advection kernel with VTK output (ensemble mode)
 *
 * @details Ensemble version with VTK output. Processes all ensemble members
 *          and optionally outputs selected ensemble for visualization.
 */

#pragma once

#include "../device/ldm_kernels_device.cuh"
#include "../cram/ldm_kernels_cram.cuh"
#include "../../physics/ldm_cram2.cuh"
#include "../../core/params.hpp"

/**
 * @kernel move_part_by_wind_mpi_ens_dump
 * @brief Particle advection with VTK output (ensemble mode)
 *
 * @details Ensemble version with VTK dump capability:
 *          - Processes all ensemble members
 *          - Collects VTK data for selected ensemble (typically ensemble 7)
 *          - Identical physics to non-dump ensemble version
 *
 *          Physics (same as move_part_by_wind_mpi_ens()):
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
 * @param[in]     mesh_nx                        VTK mesh size X
 * @param[in]     mesh_ny                        VTK mesh size Y
 * @param[in]     device_meteorological_flex_unis0  Meteorology time0 (2D)
 * @param[in]     device_meteorological_flex_pres0  Meteorology time0 (3D)
 * @param[in]     device_meteorological_flex_unis1  Meteorology time1 (2D)
 * @param[in]     device_meteorological_flex_pres1  Meteorology time1 (3D)
 * @param[in]     total_particles                Total particles (all ensembles)
 *
 * @grid_config
 *   - Block size: 256 threads
 *   - Grid size: (total_particles + 255) / 256
 *
 * @performance
 *   - Slightly slower than non-dump version (~5-10% overhead)
 *   - VTK output only for selected ensemble reduces overhead
 *
 * @note Only used for final iteration or when enable_vtk_output = true
 * @note Typical usage: Output ensemble #7 for visualization
 *
 * @see move_part_by_wind_mpi_ens() for non-dump version
 * @see move_part_by_wind_mpi_dump() for single-mode dump version
 */
__global__ void move_part_by_wind_mpi_ens_dump(
    LDM::LDMpart* d_part, float t0, int rank, float* d_dryDep, float* d_wetDep, int mesh_nx, int mesh_ny,
    FlexUnis* device_meteorological_flex_unis0,
    FlexPres* device_meteorological_flex_pres0,
    FlexUnis* device_meteorological_flex_unis1,
    FlexPres* device_meteorological_flex_pres1,
    int total_particles,
    const KernelScalars ks);

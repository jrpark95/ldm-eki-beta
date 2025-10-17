/**
 * @file ldm_kernels_dump.cuh
 * @brief Particle advection kernel with VTK output (single mode)
 *
 * @details Identical to ldm_kernels_particle.cuh but with added VTK dump
 *          functionality for visualization. Used for final iteration output.
 */

#pragma once

#include "../device/ldm_kernels_device.cuh"
#include "../cram/ldm_kernels_cram.cuh"
#include "../../physics/ldm_cram2.cuh"
#include "../../core/params.hpp"

/**
 * @kernel move_part_by_wind_mpi_dump
 * @brief Particle advection with VTK output (single mode)
 *
 * @details Same physics as move_part_by_wind_mpi() but with VTK data collection:
 *          - Accumulates particle positions to grid cells
 *          - Computes concentration fields for visualization
 *          - Writes data suitable for ParaView output
 *
 *          Physics (identical to non-dump version):
 *          - Wind advection + turbulent diffusion
 *          - Gravitational settling
 *          - Wet/dry deposition
 *          - Radioactive decay
 *
 * @param[in,out] d_part                         Particle array
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
 *
 * @grid_config
 *   - Block size: 256 threads
 *   - Grid size: (d_nop + 255) / 256
 *
 * @performance
 *   - Slightly slower than non-dump version (~5-10% overhead)
 *   - VTK accumulation uses atomic operations
 *
 * @note Only used when VTK output is enabled
 * @note Typically called for final iteration or selected timesteps
 *
 * @see move_part_by_wind_mpi() for non-dump version
 * @see move_part_by_wind_mpi_ens_dump() for ensemble dump version
 */
__global__ void move_part_by_wind_mpi_dump(
    LDM::LDMpart* d_part, float t0, int rank, float* d_dryDep, float* d_wetDep, int mesh_nx, int mesh_ny,
    FlexUnis* device_meteorological_flex_unis0,
    FlexPres* device_meteorological_flex_pres0,
    FlexUnis* device_meteorological_flex_unis1,
    FlexPres* device_meteorological_flex_pres1,
    const KernelScalars ks);

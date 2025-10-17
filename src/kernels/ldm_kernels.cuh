/**
 * @file ldm_kernels.cuh
 * @brief Master include file for all LDM CUDA kernels
 *
 * @details This file provides a single include point for all LDM kernel modules.
 *          The original monolithic ldm_kernels.cuh (3,864 lines) has been split
 *          into logical modules for better maintainability and compilation speed.
 *
 * @note Original file backed up as ldm_kernels.cuh.ORIGINAL_BACKUP
 *
 * Module organization:
 *   - device/    : Device utility functions (nuclear decay, meteorology, RNG)
 *   - particle/  : Particle movement kernels (single and ensemble modes)
 *   - eki/       : EKI observation system (receptor dose computation)
 *   - dump/      : Particle movement with VTK output
 *
 * Total: ~3,864 lines split into 14 files (avg ~275 lines each)
 */

#pragma once

// ============================================================================
// DEVICE UTILITY FUNCTIONS (~190 lines)
// ============================================================================
// Provides: nuclear_decay_optimized_inline, k_f_esi, k_f_esl, k_f_qvsat,
//           getRand, GaussianRand, init_curand_states, update_particle_flags,
//           update_particle_flags_ens, Dynamic_viscosity
#include "device/ldm_kernels_device.cuh"

// ============================================================================
// EKI OBSERVATION SYSTEM (~107 lines)
// ============================================================================
// Provides: compute_eki_receptor_dose, compute_eki_receptor_dose_ensemble
#include "eki/ldm_kernels_eki.cuh"

// ============================================================================
// PARTICLE MOVEMENT KERNELS (~893 + 903 lines)
// ============================================================================
// Provides: move_part_by_wind_mpi (single mode)
//           move_part_by_wind_mpi_ens (ensemble mode)
#include "particle/ldm_kernels_particle.cuh"
#include "particle/ldm_kernels_particle_ens.cuh"

// ============================================================================
// PARTICLE MOVEMENT WITH VTK OUTPUT (~884 + 889 lines)
// ============================================================================
// Provides: move_part_by_wind_mpi_dump (single mode with VTK)
//           move_part_by_wind_mpi_ens_dump (ensemble mode with VTK)
#include "dump/ldm_kernels_dump.cuh"
#include "dump/ldm_kernels_dump_ens.cuh"

// ============================================================================
// COMPILATION NOTES
// ============================================================================
//
// Benefits of the new structure:
//   1. Parallel compilation: 14 small files instead of 1 large file
//   2. Better organization: Logical grouping by functionality
//   3. Easier maintenance: ~275 lines per file vs 3,864 lines
//   4. Incremental builds: Changes only recompile affected modules
//
// Migration from old code:
//   - All function signatures unchanged
//   - Same physics, same algorithms
//   - Only file organization changed
//   - Original backed up as ldm_kernels.cuh.ORIGINAL_BACKUP
//
// ============================================================================

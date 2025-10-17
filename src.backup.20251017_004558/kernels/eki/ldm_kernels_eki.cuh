/**
 * @file ldm_kernels_eki.cuh
 * @brief EKI observation system kernels for receptor dose computation
 *
 * @details This module implements the EKI (Ensemble Kalman Inversion) observation
 *          kernels that compute receptor measurements based on particle positions.
 *          Supports both single-mode and ensemble-mode simulations.
 */

#pragma once
#include "../../core/ldm.cuh"  // For LDM::LDMpart and constants
#include "../../core/params.hpp"  // For KernelScalars

// ============================================================================
// EKI OBSERVATION KERNELS
// ============================================================================

/**
 * @kernel compute_eki_receptor_dose
 * @brief Compute receptor dose measurements from particle positions (single mode)
 *
 * @details Processes particles to calculate dose contributions at receptor locations.
 *          Each thread handles one particle, checking distance to all receptors.
 *          Uses atomic operations for thread-safe accumulation.
 *
 * @param[in]     particles                Particle array
 * @param[in]     receptor_lats            Receptor latitudes [degrees]
 * @param[in]     receptor_lons            Receptor longitudes [degrees]
 * @param[in]     receptor_capture_radius  Capture radius [degrees]
 * @param[in,out] receptor_dose            Dose output [timestep][receptor]
 * @param[in,out] receptor_particle_count  Particle count [timestep][receptor]
 * @param[in]     num_receptors            Number of receptors
 * @param[in]     num_timesteps            Number of time steps
 * @param[in]     time_idx                 Current timestep index
 * @param[in]     DCF                      Dose conversion factor (default: 1.0)
 *
 * @grid_config
 *   - Block size: 256 threads
 *   - Grid size: (d_nop + 255) / 256
 *
 * @performance
 *   - Memory access: Coalesced particle reads
 *   - Synchronization: Atomic additions for dose accumulation
 *
 * @invariants
 *   - Skips inactive particles (flag == 0)
 *   - Ignores particles above 5000m altitude
 *   - Preserves total mass in dose calculation
 *
 * @note Dose index: time_idx * num_receptors + r (row-major order)
 * @note Coordinates: Grid to lat/lon: lat = -90 + y*0.5, lon = -179 + x*0.5
 *
 * @see compute_eki_receptor_dose_ensemble() for ensemble version
 */
__global__ void compute_eki_receptor_dose(
    const LDM::LDMpart* particles,
    const float* receptor_lats, const float* receptor_lons,
    float receptor_capture_radius,
    float* receptor_dose,  // 2D: [num_receptors * num_timesteps]
    int* receptor_particle_count,  // 2D: [num_receptors * num_timesteps]
    int num_receptors,
    int num_timesteps,
    int time_idx,  // Which timestep slot to accumulate into
    int num_particles,
    float simulation_time_end,
    float DCF,
    const KernelScalars ks);

/**
 * @kernel compute_eki_receptor_dose_ensemble
 * @brief Compute receptor dose measurements for all ensemble members
 *
 * @details Ensemble-mode version that processes particles from multiple
 *          ensemble members simultaneously. Uses particle.ensemble_id
 *          to route doses to correct ensemble slot.
 *
 * @param[in]     particles                Particle array (all ensembles)
 * @param[in]     receptor_lats            Receptor latitudes [degrees]
 * @param[in]     receptor_lons            Receptor longitudes [degrees]
 * @param[in]     receptor_capture_radius  Capture radius [degrees]
 * @param[in,out] ensemble_dose            Dose [ensemble][timestep][receptor]
 * @param[in,out] ensemble_particle_count  Count [ensemble][timestep][receptor]
 * @param[in]     num_ensembles            Number of ensemble members
 * @param[in]     num_receptors            Number of receptors
 * @param[in]     num_timesteps            Number of time steps
 * @param[in]     time_idx                 Current timestep index
 * @param[in]     total_particles          Total particles across all ensembles
 * @param[in]     DCF                      Dose conversion factor (default: 1.0)
 *
 * @grid_config
 *   - Block size: 256 threads
 *   - Grid size: (total_particles + 255) / 256
 *
 * @performance
 *   - Memory access: Coalesced particle reads
 *   - Synchronization: Atomic additions per ensemble
 *
 * @invariants
 *   - Maintains separate accumulation for each ensemble
 *   - Skips inactive particles (flag == 0)
 *   - Ignores particles above 5000m altitude
 *
 * @note Index formula: ens_id * (TIME * RECEPT) + time_idx * RECEPT + r
 * @note Matches reference implementation for IPC communication
 *
 * @see compute_eki_receptor_dose() for single-mode version
 */
__global__ void compute_eki_receptor_dose_ensemble(
    const LDM::LDMpart* particles,
    const float* receptor_lats, const float* receptor_lons,
    float receptor_capture_radius,
    float* ensemble_dose,  // [num_ensembles × num_receptors × num_timesteps]
    int* ensemble_particle_count,  // [num_ensembles × num_receptors × num_timesteps]
    int num_ensembles,
    int num_receptors,
    int num_timesteps,
    int time_idx,  // Which timestep slot to accumulate into
    int total_particles,
    int particles_per_ensemble,
    float simulation_time_end,
    float DCF,
    const KernelScalars ks);

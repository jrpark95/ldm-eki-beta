/**
 * @file ldm_kernels_eki.cu
 * @brief Implementation of EKI observation system kernels
 *
 * @details Implements receptor dose computation for Ensemble Kalman Inversion.
 *          Processes particles to calculate gamma dose measurements at
 *          specified receptor locations. Supports both single-mode (initial
 *          "true" simulation) and ensemble-mode (EKI iterations).
 *
 * @author Juryong Park, 2025
 */

#include "ldm_kernels_eki.cuh"

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * @device safe_div
 * @brief Safe division helper to prevent NaN from division by zero
 *
 * @details Returns zero if denominator is non-positive, otherwise performs
 *          normal division. Used for dose calculations where zero particle
 *          count should result in zero dose.
 *
 * @param[in] num  Numerator
 * @param[in] den  Denominator
 *
 * @return num/den if den > 0, otherwise 0.0f
 *
 * @complexity O(1)
 * @note Inlined for zero overhead
 */
__device__ __forceinline__ float safe_div(float num, float den) {
    return (den > 0.0f) ? (num / den) : 0.0f;
}

// ============================================================================
// EKI OBSERVATION SYSTEM IMPLEMENTATION
// ============================================================================

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
    const KernelScalars ks) {

    // Thread-to-particle mapping: one thread per particle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // Load particle data (coalesced read)
    LDM::LDMpart particle = particles[idx];
    if (particle.flag == 0) return; // Skip inactive particles (not yet emitted or deposited)

    // Convert grid indices to geographic coordinates
    // Grid: x ∈ [0, 720] (0.5° resolution), y ∈ [0, 360]
    // Geographic: lon ∈ [-179, +180], lat ∈ [-90, +90]
    float lat = -90.0f + particle.y * 0.5f;
    float lon = -179.0f + particle.x * 0.5f;

    // Exclude particles at unrealistic altitudes (above typical PBL + free troposphere)
    // 5000m is a practical cutoff for ground-level dose contributions
    if (particle.z > 5000.0f) return;

    // Loop over all receptors to check proximity
    // Note: This is O(num_receptors) per particle, acceptable for ~10 receptors
    for (int r = 0; r < num_receptors; ++r) {
        float receptor_lat = receptor_lats[r];
        float receptor_lon = receptor_lons[r];

        // Simple rectangular bounding box check (faster than Haversine distance)
        // receptor_capture_radius is in degrees (e.g., 0.5° ≈ 55 km at equator)
        float lat_diff = fabs(lat - receptor_lat);
        float lon_diff = fabs(lon - receptor_lon);

        if (lat_diff <= receptor_capture_radius && lon_diff <= receptor_capture_radius) {
            // Compute dose contribution from this particle
            // Formula: dose = (concentration × DCF × time) / num_particles
            //   - concentration: nuclide activity [Bq]
            //   - DCF: dose conversion factor [Sv/Bq]
            //   - time: integration time [s]
            //   - num_particles: normalization for statistical sampling
            float dose_increment = safe_div(
                particle.conc * DCF * simulation_time_end,
                static_cast<float>(num_particles)
            );

            // Compute output index in row-major order: [timestep][receptor]
            // This matches the Python IPC reader's expectation:
            //   gamma_dose_matrix[time_idx, receptor_idx]
            int dose_idx = time_idx * num_receptors + r;

            // Atomic accumulation (thread-safe, required for multiple particles hitting same receptor)
            // Performance note: Atomic adds can cause contention, but acceptable for ~10 receptors
            atomicAdd(&receptor_dose[dose_idx], dose_increment);
            atomicAdd(&receptor_particle_count[dose_idx], 1);
        }
    }
}

__global__ void compute_eki_receptor_dose_ensemble(
    const LDM::LDMpart* particles,
    const float* receptor_lats, const float* receptor_lons,
    float receptor_capture_radius,
    float* ensemble_dose,  // [num_ensembles × num_receptors × num_timesteps]
    int* ensemble_particle_count,  // [num_ensembles × num_receptors × num_timesteps]
    int num_ensembles,
    int num_receptors,
    int num_timesteps,
    int time_idx,  // Which timestep slot to accumulate into (same as single mode)
    int total_particles,
    int particles_per_ensemble,
    float simulation_time_end,
    float DCF,
    const KernelScalars ks) {

    // Thread-to-particle mapping: one thread per particle (across ALL ensembles)
    // Example: If 100 ensembles × 10,000 particles = 1,000,000 total threads
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_particles) return;

    // Load particle data (coalesced read)
    LDM::LDMpart particle = particles[idx];
    if (particle.flag == 0) return; // Skip inactive particles

    // Extract ensemble membership from particle metadata
    // Each particle knows which ensemble it belongs to (0 to num_ensembles-1)
    int ens_id = particle.ensemble_id;

    // Convert grid indices to geographic coordinates
    // Same coordinate system as single mode
    float lat = -90.0f + particle.y * 0.5f;
    float lon = -179.0f + particle.x * 0.5f;

    // Exclude high-altitude particles (same as single mode)
    if (particle.z > 5000.0f) return;

    // Loop over all receptors to check proximity
    for (int r = 0; r < num_receptors; ++r) {
        float receptor_lat = receptor_lats[r];
        float receptor_lon = receptor_lons[r];

        // Rectangular bounding box check (same as single mode)
        float lat_diff = fabs(lat - receptor_lat);
        float lon_diff = fabs(lon - receptor_lon);

        if (lat_diff <= receptor_capture_radius && lon_diff <= receptor_capture_radius) {
            // Compute dose contribution from this particle
            // Normalization uses particles_per_ensemble (NOT total_particles)
            // because each ensemble is independent
            float dose_increment = safe_div(
                particle.conc * DCF * simulation_time_end,
                static_cast<float>(particles_per_ensemble)
            );

            // Compute 3D output index: [ensemble][timestep][receptor]
            // Layout matches Python IPC reader's expectation:
            //   ensemble_observations[ens_id, time_idx, receptor_idx]
            // Formula: ens_id × (TIME × RECEPT) + time_idx × RECEPT + r
            int output_idx = ens_id * (num_timesteps * num_receptors) +
                           time_idx * num_receptors +
                           r;

            // Atomic accumulation for this ensemble's dose array
            // Each ensemble has its own separate accumulation buffer
            atomicAdd(&ensemble_dose[output_idx], dose_increment);
            atomicAdd(&ensemble_particle_count[output_idx], 1);
        }
    }
}

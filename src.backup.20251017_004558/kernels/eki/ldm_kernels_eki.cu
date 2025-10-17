/**
 * @file ldm_kernels_eki.cu
 * @brief Implementation of EKI observation system kernels
 */

#include "ldm_kernels_eki.cuh"

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * @brief Safe division helper to prevent NaN from division by zero
 * @param num Numerator
 * @param den Denominator
 * @return num/den if den > 0, otherwise 0.0f
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

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    LDM::LDMpart particle = particles[idx];
    if (particle.flag == 0) return; // Skip inactive particles

    // Convert grid coordinates to lat/lon
    float lat = -90.0f + particle.y * 0.5f;
    float lon = -179.0f + particle.x * 0.5f;

    // Skip particles outside reasonable altitude range
    if (particle.z > 5000.0f) return;

    // Check each receptor
    for (int r = 0; r < num_receptors; ++r) {
        float receptor_lat = receptor_lats[r];
        float receptor_lon = receptor_lons[r];

        // Check if particle is within receptor capture radius
        float lat_diff = fabs(lat - receptor_lat);
        float lon_diff = fabs(lon - receptor_lon);

        if (lat_diff <= receptor_capture_radius && lon_diff <= receptor_capture_radius) {
            // Calculate dose contribution from this particle (with safe division to prevent NaN)
            float dose_increment = safe_div(
                particle.conc * DCF * simulation_time_end,
                static_cast<float>(num_particles)
            );

            // Calculate 2D index: [timestep][receptor] - MATCH REFERENCE CODE
            // Reference: gamma_dose_idx = ens * (TIME * RECEPT) + r + time_idx * RECEPT
            // Single mode: just time_idx * num_receptors + r
            int dose_idx = time_idx * num_receptors + r;

            // Atomically add to the correct time_idx slot
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

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_particles) return;

    LDM::LDMpart particle = particles[idx];
    if (particle.flag == 0) return; // Skip inactive particles

    // Get ensemble ID from particle
    int ens_id = particle.ensemble_id;

    // Convert grid coordinates to lat/lon
    float lat = -90.0f + particle.y * 0.5f;
    float lon = -179.0f + particle.x * 0.5f;

    // Skip particles outside reasonable altitude range
    if (particle.z > 5000.0f) return;

    // Check each receptor
    for (int r = 0; r < num_receptors; ++r) {
        float receptor_lat = receptor_lats[r];
        float receptor_lon = receptor_lons[r];

        // Check if particle is within receptor capture radius
        float lat_diff = fabs(lat - receptor_lat);
        float lon_diff = fabs(lon - receptor_lon);

        if (lat_diff <= receptor_capture_radius && lon_diff <= receptor_capture_radius) {
            // Calculate dose contribution from this particle (with safe division to prevent NaN)
            float dose_increment = safe_div(
                particle.conc * DCF * simulation_time_end,
                static_cast<float>(particles_per_ensemble)
            );

            // Calculate output index: [ensemble][timestep][receptor] - MATCH REFERENCE CODE
            // Reference: gamma_dose_idx = ens * (TIME * RECEPT) + r + time_idx * RECEPT
            int output_idx = ens_id * (num_timesteps * num_receptors) +
                           time_idx * num_receptors +
                           r;

            // Atomically add to the correct time_idx slot for this ensemble
            atomicAdd(&ensemble_dose[output_idx], dose_increment);
            atomicAdd(&ensemble_particle_count[output_idx], 1);
        }
    }
}

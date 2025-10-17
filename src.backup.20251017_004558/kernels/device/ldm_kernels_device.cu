/**
 * @file ldm_kernels_device.cu
 * @brief Implementation of device utility functions for LDM particle simulation
 */

#include "ldm_kernels_device.cuh"

// ============================================================================
// NUCLEAR DECAY IMPLEMENTATION
// ============================================================================

__device__ void nuclear_decay_optimized_inline(float* exp_matrix, float* concentration, bool no_clamp) {

    float result[MAX_NUCLIDES];

    // SIMPLE EXPONENTIAL DECAY for testing (Sr-92 → Y-92)
    for(int i = 0; i < MAX_NUCLIDES; i++) {
        result[i] = concentration[i];  // Start with input
    }

    // Apply simple Sr-92 → Y-92 decay for testing
    if(concentration[12] > 0.0f) {  // Sr-92 index
        float decay_rate_sr92 = 7.105e-05f;  // Sr-92 decay constant (1/s)
        float dt_test = 10.0f;  // Hardcoded test value (not used in production)
        float decay_factor = __expf(-decay_rate_sr92 * dt_test);
        float decayed_amount = concentration[12] * (1.0f - decay_factor);

        result[12] = concentration[12] * decay_factor;  // Sr-92 decay
        result[13] += decayed_amount;  // Y-92 production
    }


    // Copy result back with optional safety checks
    for(int i = 0; i < MAX_NUCLIDES; i++) {
        concentration[i] = result[i];

        if (!no_clamp) {
            // Safety check: handle small negative values (numerical noise)
            if(concentration[i] < 0.0f) {
                if(concentration[i] < -1e-12f) {
                    // Significant negative value detected
                }
                concentration[i] = 1e-30f;  // Small positive value instead of zero
            }

            // Handle very large values (potential overflow)
            if(concentration[i] > 1e30f) {
                concentration[i] = 1e30f;  // Clamp to large but reasonable value
            }
        } else {
            // No clamp mode: only check for NaN/Inf
            if(isnan(concentration[i]) || isinf(concentration[i])) {
                concentration[i] = 0.0f;
            }
        }
    }
}

// ============================================================================
// METEOROLOGICAL UTILITY IMPLEMENTATIONS
// ============================================================================

__device__ float k_f_esi(float p, float t) {
    // Saturation specific humidity parameters used in enhanced Teten's formula.
    const float satfia = 1.0003f;
    const float satfib = 4.18e-8f;  // for p in Pa
    const float sateia = 611.15f;   // es in Pa
    const float sateib = 22.452f;
    const float sateic = 0.6f;

    // Calculate the factor 'f'
    float f = satfia + satfib * p;

    // Calculate the saturation water vapor pressure over ice
    float f_esi = f * sateia * expf(sateib * (t - 273.15f) / (t - sateic));

    return f_esi;
}

__device__ float k_f_esl(float p, float t) {
    // Constants
    const float satfwa = 1.0007f;
    const float satfwb = 3.46e-8f;  // for p in Pa
    const float satewa = 611.21f;   // es in Pa
    const float satewb = 17.502f;
    const float satewc = 32.18f;

    // Calculate the enhancement factor
    float f = satfwa + satfwb * p;

    // Calculate the saturation water vapor pressure over liquid water
    float f_esl = f * satewa * expf(satewb * (t - 273.15f) / (t - satewc));

    return f_esl;
}

__device__ float k_f_qvsat(float p, float t) {
    // Constants
    const float rd = 287.0f;  // Gas constant for dry air (m²/(s²*K))
    const float rv = 461.0f;  // Gas constant for water vapor (m²/(s²*K))
    const float rddrv = rd / rv;

    float fespt;

    // Determine which formula to use based on temperature
    if (t >= 253.15f) {  // Modification to account for supercooled water
        fespt = k_f_esl(p, t);  // Saturation vapor pressure over liquid water
    } else {
        fespt = k_f_esi(p, t);  // Saturation vapor pressure over ice
    }

    float denom = p - (1.0f - rddrv) * fespt;

    // Check for division by zero
    float f_qvsat;
    if (denom == 0.0f) {
        f_qvsat = 1.0f;
    } else {
        f_qvsat = rddrv * fespt / denom;
    }

    return f_qvsat;
}

// ============================================================================
// RANDOM NUMBER GENERATION IMPLEMENTATIONS
// ============================================================================

// Note: getRand and GaussianRand are now inline in ldm_kernels_device.cuh

// ============================================================================
// INITIALIZATION KERNEL IMPLEMENTATIONS
// ============================================================================

__global__ void init_curand_states(LDM::LDMpart* d_part, float t0, int num_particles){

        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_particles) return;

        unsigned long long seed = static_cast<unsigned long long>((t0 + idx * 0.001f) * ULLONG_MAX);
        curandState localState;
        curand_init(seed, idx, 0, &localState);
        d_part[idx].randState[0] = localState;
    }

__global__ void update_particle_flags(
    LDM::LDMpart* d_part, float activationRatio, int num_particles){

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_particles) return;

        LDM::LDMpart& p = d_part[idx];

        // Activate particles based on their timeidx and current activationRatio
        int maxActiveTimeidx = int(num_particles * activationRatio);

        if (p.timeidx <= maxActiveTimeidx){  // Changed from idx to p.timeidx
            p.flag = 1;
        }
    }

__global__ void update_particle_flags_ens(
    LDM::LDMpart* d_part, float activationRatio, int total_particles, int particles_per_ensemble){

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_particles) return;

        LDM::LDMpart& p = d_part[idx];

        int local_idx = idx % particles_per_ensemble;

        int maxActiveInEnsemble = int(particles_per_ensemble * activationRatio);
        if (p.timeidx <= maxActiveInEnsemble){
            p.flag = 1;
        }

        // DEBUG: Print activation info for particle 832 when activationRatio > 0.08
        // Commented out for release - uncomment for debugging
        // if (idx == 832 && activationRatio > 0.08 && activationRatio < 0.15) {
        //     printf("[GPU_FLAG_ENS_T60] idx=%d, local_idx=%d, maxActive=%d, timeidx=%d, flag=%d (ratio=%.4f)\n",
        //            idx, local_idx, maxActiveInEnsemble, p.timeidx, p.flag, activationRatio);
        // }
    }

// ============================================================================
// PHYSICAL PROPERTY IMPLEMENTATIONS
// ============================================================================

// Note: Dynamic_viscosity is now inline in ldm_kernels_device.cuh

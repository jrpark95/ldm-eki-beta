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
    // NOTE: This is a TEST FUNCTION not used in production
    // Production code uses apply_T_once_rowmajor_60() for full decay chains
    for(int i = 0; i < MAX_NUCLIDES; i++) {
        result[i] = concentration[i];  // Start with input concentrations
    }

    // Apply simple two-nuclide decay chain: Sr-92 → Y-92
    if(concentration[12] > 0.0f) {  // Sr-92 index in nuclide array
        // Sr-92 decay constant: 7.105e-05 s^-1 (half-life ~2.71 hours)
        float decay_rate_sr92 = 7.105e-05f;
        float dt_test = 10.0f;  // Hardcoded test timestep (not used in production)

        // Compute decay factor: exp(-λt)
        float decay_factor = __expf(-decay_rate_sr92 * dt_test);
        float decayed_amount = concentration[12] * (1.0f - decay_factor);

        result[12] = concentration[12] * decay_factor;  // Sr-92 remaining
        result[13] += decayed_amount;  // Y-92 production (daughter product)
    }


    // Copy results back with optional safety checks for numerical stability
    for(int i = 0; i < MAX_NUCLIDES; i++) {
        concentration[i] = result[i];

        if (!no_clamp) {
            // Safety mode: clamp to reasonable ranges

            // Handle small negative values from floating-point round-off
            if(concentration[i] < 0.0f) {
                if(concentration[i] < -1e-12f) {
                    // Significant negative value detected (potential algorithm issue)
                }
                concentration[i] = 1e-30f;  // Replace with small positive value
            }

            // Handle very large values (potential overflow or divergence)
            if(concentration[i] > 1e30f) {
                concentration[i] = 1e30f;  // Clamp to large but representable value
            }
        } else {
            // No-clamp mode: only check for NaN/Inf (invalid floating-point)
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
    // Enhanced Teten's formula for saturation vapor pressure over ice
    // Reference: Teten (1930), Buck (1981) enhancement factor
    const float satfia = 1.0003f;      // Enhancement factor coefficient
    const float satfib = 4.18e-8f;     // Pressure-dependent enhancement [Pa^-1]
    const float sateia = 611.15f;      // Reference saturation vapor pressure [Pa]
    const float sateib = 22.452f;      // Teten coefficient for ice
    const float sateic = 0.6f;         // Temperature offset [K]

    // Enhancement factor accounts for non-ideal gas behavior at high pressure
    float f = satfia + satfib * p;

    // Teten's exponential formula: es = f * es0 * exp(a*(T-T0)/(T-b))
    float f_esi = f * sateia * expf(sateib * (t - 273.15f) / (t - sateic));

    return f_esi;
}

__device__ float k_f_esl(float p, float t) {
    // Enhanced Teten's formula for saturation vapor pressure over liquid water
    // Reference: Teten (1930), Buck (1981) enhancement factor
    const float satfwa = 1.0007f;      // Enhancement factor coefficient
    const float satfwb = 3.46e-8f;     // Pressure-dependent enhancement [Pa^-1]
    const float satewa = 611.21f;      // Reference saturation vapor pressure [Pa]
    const float satewb = 17.502f;      // Teten coefficient for liquid water
    const float satewc = 32.18f;       // Temperature offset [K]

    // Enhancement factor for liquid water (slightly different from ice)
    float f = satfwa + satfwb * p;

    // Teten's formula with enhancement factor
    float f_esl = f * satewa * expf(satewb * (t - 273.15f) / (t - satewc));

    return f_esl;
}

__device__ float k_f_qvsat(float p, float t) {
    // Calculate saturation mixing ratio from Clausius-Clapeyron relation
    const float rd = 287.0f;           // Specific gas constant for dry air [J/(kg·K)]
    const float rv = 461.0f;           // Specific gas constant for water vapor [J/(kg·K)]
    const float rddrv = rd / rv;       // Ratio ≈ 0.622 (molecular weight ratio)

    float fespt;

    // Select appropriate saturation formula based on temperature
    // 253.15 K (-20°C) threshold accounts for supercooled liquid water
    if (t >= 253.15f) {
        fespt = k_f_esl(p, t);  // Use liquid water formula (warmer)
    } else {
        fespt = k_f_esi(p, t);  // Use ice formula (colder)
    }

    // Mixing ratio formula: qvsat = (rd/rv) * (es / (p - es))
    // Rearranged to: qvsat = (rd/rv) * es / (p - (1 - rd/rv) * es)
    float denom = p - (1.0f - rddrv) * fespt;

    // Safety check to prevent division by zero
    // (occurs only in extreme conditions where es ≈ p)
    float f_qvsat;
    if (denom == 0.0f) {
        f_qvsat = 1.0f;  // Saturated atmosphere (100% relative humidity)
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

__global__ void initCurandStates(LDM::LDMpart* d_part, float t0, int num_particles){

        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_particles) return;

        // Create unique seed for each particle using simulation time and particle index
        // Multiplying by ULLONG_MAX ensures full range of random sequence space
        unsigned long long seed = static_cast<unsigned long long>((t0 + idx * 0.001f) * ULLONG_MAX);

        // Initialize cuRAND state for this particle
        // Arguments: (seed, sequence, offset, state)
        curandState localState;
        curand_init(seed, idx, 0, &localState);

        // Store initialized state in particle structure
        d_part[idx].randState[0] = localState;
    }

__global__ void updateParticleFlags(
    LDM::LDMpart* d_part, float activationRatio, int num_particles){

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_particles) return;

        LDM::LDMpart& p = d_part[idx];

        // Progressive particle activation based on emission timeseries
        // Particles with lower timeidx values represent earlier emissions
        // activationRatio ∈ [0,1] controls what fraction of particles are active
        int maxActiveTimeidx = int(num_particles * activationRatio);

        if (p.timeidx <= maxActiveTimeidx){
            p.flag = 1;  // Activate particle (eligible for advection)
        }
    }

__global__ void updateParticleFlagsEnsemble(
    LDM::LDMpart* d_part, float activationRatio, int total_particles, int particles_per_ensemble){

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_particles) return;

        LDM::LDMpart& p = d_part[idx];

        // Compute local index within this ensemble member
        // Example: If idx=832 and particles_per_ensemble=1000, local_idx=832
        int local_idx = idx % particles_per_ensemble;

        // Activate particles progressively within each ensemble member
        // Each ensemble has its own independent activation based on timeidx
        int maxActiveInEnsemble = int(particles_per_ensemble * activationRatio);
        if (p.timeidx <= maxActiveInEnsemble){
            p.flag = 1;  // Activate this particle
        }

        // DEBUG: Print activation info for particle 832 when activationRatio > 0.08
        // Commented out for release - uncomment for debugging ensemble activation
        // if (idx == 832 && activationRatio > 0.08 && activationRatio < 0.15) {
        //     printf("[GPU_FLAG_ENS_T60] idx=%d, local_idx=%d, maxActive=%d, timeidx=%d, flag=%d (ratio=%.4f)\n",
        //            idx, local_idx, maxActiveInEnsemble, p.timeidx, p.flag, activationRatio);
        // }
    }

// ============================================================================
// PHYSICAL PROPERTY IMPLEMENTATIONS
// ============================================================================

// Note: Dynamic_viscosity is now inline in ldm_kernels_device.cuh

/**
 * @file ldm_kernels_cram.cuh
 * @brief CRAM (Chebyshev Rational Approximation Method) CUDA kernels
 *
 * @details Device functions for applying CRAM matrix transformations to
 *          nuclide concentration vectors. CRAM is used for accurate and
 *          efficient solution of radioactive decay chains.
 *
 * @note Optimized for 60-nuclide decay chains
 * @note Uses row-major matrix layout for GPU memory coalescing
 *
 * @see ldm_cram2.cuh for CRAM matrix loading
 * @see ldm_nuclides.cuh for nuclide definitions
 *
 * @author LDM-EKI Development Team
 * @date 2025-01-15
 */

#pragma once
#ifndef N_NUCLIDES
#define N_NUCLIDES 60
#endif


__device__ __forceinline__
float clamp01(float x) { return fminf(1.0f, fmaxf(0.0f, x)); }

__device__ __forceinline__
void apply_T_once_rowmajor_60(const float* __restrict__ T, float* __restrict__ conc) {
    float x[N_NUCLIDES];
    float y[N_NUCLIDES];
    

    #pragma unroll
    for (int j = 0; j < N_NUCLIDES; ++j) x[j] = conc[j];

    #pragma unroll
    for (int i = 0; i < N_NUCLIDES; ++i) {
        const float* Ti = T + i * N_NUCLIDES;
        float acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < N_NUCLIDES; ++j) {
            acc = fmaf(Ti[j], x[j], acc);
        }
        y[i] = acc;
    }

    #pragma unroll
    for (int i = 0; i < N_NUCLIDES; ++i) conc[i] = y[i];
}

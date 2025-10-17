#pragma once
// ldm_cram2.cuh - CRAM48 Matrix Exponential for Radioactive Decay Chains
#ifndef LDM_CRAM2_CUH
#define LDM_CRAM2_CUH

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include "colors.h"

#ifndef N_NUCLIDES
#define N_NUCLIDES 60
#endif

// Forward declaration of LDM class (defined in ldm.cuh)
class LDM;

/**
 * @brief CRAM48 (Chebyshev Rational Approximation Method, order 48) constants
 *
 * @details The CRAM48 method approximates the matrix exponential exp(dt*A)
 *          using 24 complex conjugate pole pairs. This provides high accuracy
 *          for stiff decay chain problems common in nuclear physics.
 *
 * @note Constants derived from Pusa & Leppänen (2010) for optimal stability
 */

/// Initial coefficient α₀ for CRAM48
extern const double ALPHA0_48;

/// Real parts of α coefficients [24 elements]
extern const double alpha_re_48[24];

/// Imaginary parts of α coefficients [24 elements]
extern const double alpha_im_48[24];

/// Real parts of θ pole locations [24 elements]
extern const double theta_re_48[24];

/// Imaginary parts of θ pole locations [24 elements]
extern const double theta_im_48[24];

/**
 * @brief Load decay matrix A from CSV file
 *
 * @details Reads N_NUCLIDES × N_NUCLIDES decay matrix from CSV format.
 *          Matrix contains decay constants with off-diagonal elements
 *          representing branching ratios.
 *
 * @param[in]  filename  Path to CSV file (e.g., "cram/A60.csv")
 * @param[out] A_matrix  Output float array [N_NUCLIDES × N_NUCLIDES]
 *
 * @return true if successful, false on file error
 *
 * @note Matrix stored in row-major order
 * @note Empty CSV cells treated as 0.0
 * @note Diagonal elements are negative decay constants (λᵢ)
 * @note Off-diagonal A[i][j] = λⱼ × branch_ratio(j→i)
 */
bool load_A_matrix(const char* filename, float* A_matrix);

// Note: LDM member functions (load_A_csv, gauss_solve_inplace,
// cram48_expm_times_ej_host, build_T_matrix_and_upload,
// initialize_cram_system) are declared in the LDM class definition
// in src/core/ldm.cuh. Implementations are in ldm_cram2.cu.

/**
 * @device cram_decay_calculation
 * @brief Apply decay transition matrix on GPU
 *
 * @details Device-side function that applies pre-computed transition matrix
 *          to particle concentration vector: n_new = T * n_old
 *
 * @param[in]     T_matrix        Transition matrix [N_NUCLIDES × N_NUCLIDES]
 * @param[in,out] concentrations  Nuclide concentrations [N_NUCLIDES]
 *
 * @note Concentrations array is modified in-place
 * @note Uses register memory for input vector (N_NUCLIDES floats)
 * @note Unrolled loops for performance
 *
 * @performance
 *   - Memory reads: N_NUCLIDES² (from constant or global memory)
 *   - FLOPs: ~2 * N_NUCLIDES² (multiply-accumulate)
 *   - Registers: N_NUCLIDES + overhead
 *
 * @cuda_usage Called from particle update kernels
 *
 * @equation n(t+dt) = exp(dt*A) * n(t) ≈ T * n(t)
 */
__device__ __forceinline__
void cram_decay_calculation(const float* __restrict__ T_matrix,
                           float* __restrict__ concentrations) {
    // Copy input to registers
    float x[N_NUCLIDES];
    #pragma unroll
    for(int i=0; i<N_NUCLIDES; i++) x[i] = concentrations[i];

    // Matrix-vector multiply: concentrations = T * x
    #pragma unroll
    for(int r=0; r<N_NUCLIDES; r++){
        float acc = 0.0f;
        const int row_off = r * N_NUCLIDES;
        #pragma unroll
        for(int c=0; c<N_NUCLIDES; c++){
            acc += T_matrix[row_off + c] * x[c];
        }
        concentrations[r] = acc;
    }
}

#endif // LDM_CRAM2_CUH

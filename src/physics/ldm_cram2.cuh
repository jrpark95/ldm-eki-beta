#pragma once
// ldm_cram2.cuh - CRAM48 Matrix Exponential for Radioactive Decay Chains
// Author: Juryong Park, 2025
#ifndef LDM_CRAM2_CUH
#define LDM_CRAM2_CUH

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include "../colors.h"

#ifndef N_NUCLIDES
#define N_NUCLIDES 60
#endif

// Forward declaration of LDM class (defined in ldm.cuh)
class LDM;

/**
 * @file ldm_cram2.cuh
 * @brief CRAM48 implementation for solving radioactive decay chains
 * @author Juryong Park
 * @date 2025
 *
 * @details This module implements the Chebyshev Rational Approximation Method
 *          (CRAM) of order 48 for computing matrix exponentials that arise in
 *          radioactive decay chain problems.
 *
 * ## Mathematical Background
 *
 * The time evolution of radioactive decay chains is governed by the Bateman
 * equations, which form a linear system of ordinary differential equations:
 *
 * @equation
 *   dn/dt = A * n
 *
 * where:
 *   - n(t) is the nuclide concentration vector [N_NUCLIDES × 1]
 *   - A is the decay matrix containing decay constants and branching ratios
 *   - A[i][i] = -λᵢ (negative decay constant)
 *   - A[i][j] = λⱼ × branch_ratio(j→i) for i ≠ j
 *
 * The formal solution is:
 *
 * @equation
 *   n(t+dt) = exp(dt * A) * n(t)
 *
 * Computing the matrix exponential directly is computationally expensive and
 * numerically unstable for stiff systems (when decay constants vary by orders
 * of magnitude). CRAM provides an efficient and stable approximation.
 *
 * ## CRAM48 Method
 *
 * The CRAM method approximates the matrix exponential using a rational function:
 *
 * @equation
 *   exp(dt * A) ≈ α₀ * I + 2 * Re[ Σₖ₌₁²⁴ αₖ * (dt*A - θₖ*I)⁻¹ ]
 *
 * where:
 *   - α₀, {αₖ}, {θₖ} are precomputed complex coefficients (24 pole pairs)
 *   - I is the identity matrix
 *   - Re[·] denotes the real part
 *   - (·)⁻¹ represents matrix inversion
 *
 * The coefficients are chosen to minimize the approximation error over the
 * left half of the complex plane, making CRAM ideal for decay problems where
 * all eigenvalues of A have negative real parts.
 *
 * @reference
 *   M. Pusa, "Rational Approximations to the Matrix Exponential in Burnup
 *   Calculations", Nuclear Science and Engineering, Vol. 169, pp. 155-167, 2011.
 *
 * @reference
 *   M. Pusa and J. Leppänen, "Computing the Matrix Exponential in Burnup
 *   Calculations", Nuclear Science and Engineering, Vol. 164, pp. 140-150, 2010.
 *
 * ## Numerical Properties
 *
 * - **Accuracy**: Relative error < 10⁻¹² for typical decay problems
 * - **Stability**: A-stable (unconditionally stable for stiff systems)
 * - **Efficiency**: O(N³) per time step (via Gaussian elimination)
 * - **Stiffness**: Handles decay constants ranging from seconds to billions of years
 *
 * ## Implementation Strategy
 *
 * 1. **Host (CPU)**: Precompute transition matrix T = exp(dt*A) once
 * 2. **Device (GPU)**: Apply T to particle concentrations via matrix-vector multiplication
 *
 * This two-stage approach exploits the fact that T depends only on the time step
 * and decay matrix, not on particle positions or concentrations. The expensive
 * matrix exponential is computed once at initialization, then reused millions
 * of times on the GPU.
 *
 * ## Usage Example
 *
 * @code
 *   // Initialize CRAM system (host-side, called once)
 *   LDM ldm;
 *   ldm.initialize_cram_system("cram/A60.csv");
 *
 *   // Apply decay on device (called per particle per timestep)
 *   __global__ void decay_kernel(float* T_matrix, float* concentrations) {
 *       cram_decay_calculation(T_matrix, concentrations);
 *   }
 * @endcode
 *
 * @see ldm_nuclides.cuh for nuclide configuration
 * @see KernelScalars::T_matrix for device-side matrix access
 */

/**
 * @name CRAM48 Coefficients
 * @brief Precomputed complex coefficients for CRAM48 method
 *
 * @details The CRAM48 method uses 24 complex conjugate pole pairs (θₖ, αₖ).
 *          These coefficients are derived from Chebyshev interpolation theory
 *          and optimized for exponential decay problems.
 *
 * @note Constants taken from Pusa (2011), Table 1
 * @note Double precision required for accurate matrix inversion
 * @note Complex conjugate pairs reduce computation (only real part needed)
 *
 * @{
 */

/// Initial scaling coefficient α₀ for CRAM48 rational approximation
extern const double ALPHA0_48;

/// Real parts of α residues [24 elements] - coefficients for partial fraction expansion
extern const double alpha_re_48[24];

/// Imaginary parts of α residues [24 elements] - used in complex conjugate pairs
extern const double alpha_im_48[24];

/// Real parts of θ poles [24 elements] - eigenvalues of the approximation
extern const double theta_re_48[24];

/// Imaginary parts of θ poles [24 elements] - locations of complex poles
extern const double theta_im_48[24];

/** @} */ // End of CRAM48 Coefficients group

/**
 * @brief Load decay matrix A from CSV file (float precision version)
 *
 * @details Reads N_NUCLIDES × N_NUCLIDES decay matrix from CSV format.
 *          The decay matrix A encodes the coupled differential equations
 *          governing radioactive decay chains.
 *
 * ## Matrix Structure
 *
 * The decay matrix A has the following structure:
 *   - **Diagonal elements**: A[i][i] = -λᵢ (negative decay constant of nuclide i)
 *   - **Off-diagonal elements**: A[i][j] = λⱼ × branch_ratio(j→i) (production rate)
 *
 * For a simple 3-nuclide chain A→B→C:
 * @code
 *   A = [ -λ₁    0     0   ]
 *       [  λ₁  -λ₂    0   ]
 *       [  0    λ₂  -λ₃  ]
 * @endcode
 *
 * For chains with branching (e.g., A→B, A→C):
 * @code
 *   A = [ -λ₁         0     0   ]
 *       [  λ₁×f₁    -λ₂    0   ]
 *       [  λ₁×f₂     0   -λ₃  ]
 * @endcode
 * where f₁ + f₂ = 1 (branching fractions must sum to 1).
 *
 * @param[in]  filename  Path to CSV file (e.g., "cram/A60.csv")
 * @param[out] A_matrix  Output float array [N_NUCLIDES × N_NUCLIDES], row-major
 *
 * @return true if successful, false on file error
 *
 * @note Matrix stored in row-major order: A[row][col] = A_matrix[row * N_NUCLIDES + col]
 * @note Empty CSV cells treated as 0.0
 * @note Rows beyond file EOF filled with zeros
 * @note Used for quick loading in simple scenarios; see LDM::load_A_csv() for double precision
 *
 * @warning Decay constants in file must be negative for diagonal elements
 *
 * @see LDM::load_A_csv() for double precision version
 * @see LDM::build_T_matrix_and_upload() for transition matrix computation
 */
bool load_A_matrix(const char* filename, float* A_matrix);

/**
 * @note LDM Member Functions
 *
 * The following functions are declared as LDM class methods in src/core/ldm.cuh
 * and implemented in ldm_cram2.cu:
 *
 * - **load_A_csv()**: Load decay matrix in double precision
 * - **gauss_solve_inplace()**: Gaussian elimination with partial pivoting
 * - **cram48_expm_times_ej_host()**: Compute one column of exp(dt*A) using CRAM48
 * - **build_T_matrix_and_upload()**: Build full transition matrix and upload to GPU
 * - **initialize_cram_system()**: Top-level initialization wrapper
 *
 * @see src/core/ldm.cuh for declarations
 * @see ldm_cram2.cu for implementations
 */

/**
 * @brief Device-side decay calculation using precomputed transition matrix
 *
 * @details This device function applies the CRAM48-computed transition matrix
 *          to update particle nuclide concentrations over one time step.
 *
 * ## Algorithm
 *
 * Performs matrix-vector multiplication:
 * @equation
 *   n_new = T * n_old
 *
 * where:
 *   - n_old: Current nuclide concentrations [N_NUCLIDES × 1]
 *   - T: Transition matrix exp(dt*A) [N_NUCLIDES × N_NUCLIDES]
 *   - n_new: Updated concentrations after time dt
 *
 * ## Optimization Strategy
 *
 * 1. **Register Caching**: Input vector copied to registers to avoid repeated reads
 * 2. **Loop Unrolling**: #pragma unroll eliminates loop overhead
 * 3. **Coalesced Memory**: T_matrix accessed in row-major order (sequential)
 * 4. **In-Place Update**: Result written directly to input array (saves memory)
 *
 * ## Memory Access Pattern
 *
 * - **Reads**: N_NUCLIDES² elements from T_matrix (global memory)
 * - **Writes**: N_NUCLIDES elements to concentrations (global memory)
 * - **Temporary**: N_NUCLIDES floats in registers
 *
 * ## Computational Complexity
 *
 * - **FLOPs**: 2 × N_NUCLIDES² (one multiply, one add per matrix element)
 * - **Time**: O(N²) - dominated by matrix-vector multiply
 * - **Register pressure**: ~N_NUCLIDES registers + loop variables
 *
 * @param[in]     T_matrix        Precomputed transition matrix T = exp(dt*A),
 *                                row-major layout [N_NUCLIDES × N_NUCLIDES]
 * @param[in,out] concentrations  Nuclide concentrations [N_NUCLIDES],
 *                                updated in-place
 *
 * @pre T_matrix must be computed by LDM::build_T_matrix_and_upload()
 * @pre concentrations must be valid nuclide activity/concentration vector
 * @post concentrations[i] = Σⱼ T[i][j] × concentrations_old[j]
 *
 * @note This function is __forceinline__ to eliminate call overhead
 * @note Called millions of times per simulation (once per particle per timestep)
 * @note Thread-safe: Each thread operates on its own concentrations array
 *
 * @performance For N_NUCLIDES=60 (typical):
 *   - 7,200 FLOPs per call
 *   - ~60 registers per thread
 *   - ~9.6 KB L1 cache per thread (T_matrix reads)
 *
 * @cuda_usage
 * @code
 *   __global__ void particle_update_kernel(KernelScalars ks, Particle* particles) {
 *       int idx = blockIdx.x * blockDim.x + threadIdx.x;
 *       if (idx < num_particles) {
 *           // Apply radioactive decay
 *           cram_decay_calculation(ks.T_matrix, particles[idx].concentrations);
 *       }
 *   }
 * @endcode
 *
 * @see LDM::build_T_matrix_and_upload() for transition matrix computation
 * @see KernelScalars::T_matrix for device pointer access
 * @see src/kernels/particle/ldm_kernels_particle.cu for usage examples
 */
__device__ __forceinline__
void cram_decay_calculation(const float* __restrict__ T_matrix,
                           float* __restrict__ concentrations) {
    // Copy input vector to registers to avoid repeated global memory reads
    // Register memory provides ~100x faster access than global memory
    float x[N_NUCLIDES];
    #pragma unroll
    for(int i=0; i<N_NUCLIDES; i++) {
        x[i] = concentrations[i];
    }

    // Matrix-vector multiply: concentrations = T * x
    // Row-major layout ensures coalesced memory access
    #pragma unroll
    for(int r=0; r<N_NUCLIDES; r++){
        float acc = 0.0f;
        const int row_off = r * N_NUCLIDES;

        // Accumulate dot product of matrix row with input vector
        #pragma unroll
        for(int c=0; c<N_NUCLIDES; c++){
            acc += T_matrix[row_off + c] * x[c];
        }

        // Write result (in-place update)
        concentrations[r] = acc;
    }
}

#endif // LDM_CRAM2_CUH

/**
 * @file device_storage.cu
 * @brief Device storage implementation for global constant arrays (deprecated)
 *
 * @details
 * This file previously defined __device__ constant memory arrays that were
 * shared across compilation units. These arrays have been migrated to regular
 * GPU memory (cudaMalloc) to support non-RDC compilation mode.
 *
 * @history Migration timeline:
 *
 * 1. CRAM T Matrix (2025-10-16)
 *    - Previous: __device__ float T_const[N_NUCLIDES * N_NUCLIDES];
 *    - Issue: "invalid device symbol" error in non-RDC mode
 *    - Solution: Migrated to LDM::d_T_matrix via cudaMalloc()
 *    - Impact: All decay calculations now use ks.T_matrix pointer
 *    - Files affected: 4 kernel files, ldm_cram2.cu, ldm_func_simulation.cu
 *
 * 2. Flex Height Levels (2025-10-16)
 *    - Previous: __device__ float d_flex_hgt[50];
 *    - Issue: "invalid device symbol" + "illegal memory access" in EKI mode
 *    - Solution: Migrated to LDM::d_flex_hgt via cudaMalloc()
 *    - Impact: All vertical interpolation now uses ks.flex_hgt pointer
 *    - Files affected: 4 kernel files, ldm_mdata_loading.cu, ldm_mdata_cache.cu
 *
 * @architecture Old vs. New:
 *
 * Old (RDC mode):
 * @code
 *   // device_storage.cu
 *   __device__ float d_flex_hgt[50];
 *
 *   // Some other file
 *   extern __device__ float d_flex_hgt[];
 *   cudaMemcpyToSymbol(d_flex_hgt, h_data, size);
 *
 *   // Kernel
 *   __global__ void kernel() {
 *       float height = d_flex_hgt[idx];  // Direct access
 *   }
 * @endcode
 *
 * New (non-RDC mode):
 * @code
 *   // ldm.cuh
 *   class LDM {
 *       float* d_flex_hgt;
 *   };
 *
 *   // ldm.cu
 *   LDM::LDM() {
 *       cudaMalloc(&d_flex_hgt, 50 * sizeof(float));
 *   }
 *
 *   // Kernel call site
 *   ks.flex_hgt = d_flex_hgt;
 *   kernel<<<blocks, threads>>>(ks);
 *
 *   // Kernel
 *   __global__ void kernel(KernelScalars ks) {
 *       float height = ks.flex_hgt[idx];  // Pointer access
 *   }
 * @endcode
 *
 * @benefits Non-RDC mode:
 * - Smaller binaries (~30% reduction)
 * - Faster compilation (no device link stage)
 * - Better CUDA toolkit compatibility
 * - Simpler build system
 * - Easier debugging
 * - No "invalid device symbol" errors
 *
 * @note This file is retained for documentation purposes
 * @note Can be safely deleted once all references removed
 * @note Does not contribute to compilation (no actual code)
 *
 * @author Juryong Park
 * @date 2025-10-16 (Non-RDC migration)
 * @see src/core/ldm.cuh for new memory management
 * @see src/core/params.hpp for KernelScalars definition
 */

// ============================================================================
// REMOVED: All __device__ constant memory arrays
// ============================================================================
//
// This file previously contained __device__ declarations for:
// - d_flex_hgt[50]: Vertical height levels for meteorological interpolation
// - T_const[N*N]: CRAM decay transition matrix for radioactive decay chains
//
// All arrays have been migrated to regular GPU memory (cudaMalloc) and are
// now managed by the LDM class. See header file for migration details.
//
// ============================================================================

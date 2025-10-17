/**
 * @file device_storage.cuh
 * @brief Device storage header for global constant arrays (deprecated)
 *
 * @details
 * This file previously declared __device__ constant memory arrays that were
 * shared across compilation units in RDC (Relocatable Device Code) mode.
 * However, to support non-RDC compilation (required for better compatibility
 * and smaller binary sizes), all such arrays have been migrated to regular
 * GPU memory allocated via cudaMalloc().
 *
 * @history Removed arrays:
 * - d_flex_hgt: Vertical height levels (50 floats)
 *   - Now: Allocated in LDM class constructor via cudaMalloc()
 *   - Reason: __device__ symbols not accessible in non-RDC mode
 *   - Migration: 2025-10-16 (Flex height refactoring)
 *
 * - T_const: CRAM decay transition matrix (N_NUCLIDES * N_NUCLIDES floats)
 *   - Now: Allocated in LDM class as d_T_matrix via cudaMalloc()
 *   - Reason: __device__ symbols not accessible in non-RDC mode
 *   - Migration: 2025-10-16 (CRAM T matrix refactoring)
 *
 * @rationale Non-RDC compilation benefits:
 * - Smaller binary size (~30% reduction typical)
 * - Faster compilation (no device link stage)
 * - Better compatibility across CUDA toolkit versions
 * - Simpler build system (no -rdc=true flag needed)
 * - Easier debugging (symbols match source files)
 *
 * @architecture New pattern:
 * 1. Declare pointer in LDM class: float* d_array_name;
 * 2. Allocate in constructor: cudaMalloc(&d_array_name, size);
 * 3. Free in destructor: cudaFree(d_array_name);
 * 4. Pass to kernels via KernelScalars struct
 * 5. Access in kernels: ks.array_name[index]
 *
 * @note This file is retained as a placeholder to document the migration
 * @note Can be safely deleted once all code references are removed
 *
 * @author Juryong Park
 * @date 2025-10-16 (Non-RDC migration)
 */

#ifndef DEVICE_STORAGE_CUH
#define DEVICE_STORAGE_CUH

// ============================================================================
// REMOVED: d_flex_hgt
// ============================================================================
// Previously: __device__ float d_flex_hgt[50];
// Now: Allocated in LDM class via cudaMalloc()
// Location: src/core/ldm.cuh (member: float* d_flex_hgt)
// Passed via: KernelScalars::flex_hgt
// See: src/data/meteo/ldm_mdata_loading.cu (allocation)
// See: src/data/meteo/ldm_mdata_cache.cu (EKI mode allocation)
// Commit: "Fix: Resolve d_flex_hgt invalid device symbol errors"

// ============================================================================
// REMOVED: T_const (CRAM decay matrix)
// ============================================================================
// Previously: __device__ float T_const[N_NUCLIDES * N_NUCLIDES];
// Now: Allocated in LDM class via cudaMalloc()
// Location: src/core/ldm.cuh (member: float* d_T_matrix)
// Passed via: KernelScalars::T_matrix
// See: src/physics/ldm_cram2.cu (allocation)
// Commit: "Fix: CRAM T matrix refactoring - migrate from constant to regular GPU memory"

#endif // DEVICE_STORAGE_CUH

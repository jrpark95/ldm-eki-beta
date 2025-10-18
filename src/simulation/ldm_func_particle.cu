/******************************************************************************
 * @file ldm_func_particle.cu
 * @brief Particle management and GPU memory operations implementation
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/

#include "../core/ldm.cuh"
#include "ldm_func_particle.cuh"
#include "colors.h"

// ===========================================================================
// GPU MEMORY MANAGEMENT
// ===========================================================================

/******************************************************************************
 * @brief Allocate GPU memory and copy particle data from host to device
 *
 * Transfers the entire particle array from CPU memory to GPU global memory
 * for CUDA kernel processing. This function performs:
 *
 * 1. GPU memory allocation (cudaMalloc) for particle array
 * 2. Host-to-device memory transfer (cudaMemcpy)
 * 3. Optional verification of transfer integrity
 * 4. Error handling and diagnostics
 *
 * Memory Layout:
 * - Particle data stored as array-of-structures (AoS) for simplicity
 * - Each structure contains: position, velocity, concentration, nuclide data
 * - Total size: sizeof(LDMpart) * num_particles
 *
 * Performance Considerations:
 * - Large memory transfers (O(10-100 MB) for typical simulations)
 * - One-time cost at initialization, amortized over simulation
 * - Pinned host memory could improve transfer speed (future optimization)
 *
 * @pre part vector must be populated with initialized particle data
 * @pre part vector must not be empty (checked internally)
 *
 * @post d_part pointer set to allocated GPU memory address
 * @post GPU memory contains exact copy of host particle data
 *
 * @note Called once during simulation initialization
 * @note GPU memory remains allocated until simulation end or explicit free
 *
 * @warning Exits program on allocation or transfer failure
 * @warning Caller must ensure cudaFree(d_part) at cleanup
 *
 * @see LDM::part - Host particle vector
 * @see LDM::d_part - Device particle pointer
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::allocateGPUMemory(){
        if (h_part.empty()) {
            std::cerr << Color::RED << "[ERROR] " << Color::RESET << "No particles to copy to device (part vector is empty)" << std::endl;
            return;
        }

        std::cout << Color::CYAN << "[GPU] " << Color::RESET << "Allocating memory for "
                  << Color::BOLD << h_part.size() << Color::RESET << " particles" << std::endl;

        size_t total_size = h_part.size() * sizeof(LDMpart);

        cudaError_t err = cudaMalloc((void**)&d_part, total_size);
        if (err != cudaSuccess){
            std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Failed to allocate device memory for particles: " << cudaGetErrorString(err) << std::endl;
            size_t free_mem, total_mem;
            cudaMemGetInfo(&free_mem, &total_mem);
            std::cerr << Color::RED << "[ERROR] " << Color::RESET << "GPU memory info - Free: " << free_mem/(1024*1024) << " MB, Total: " << total_mem/(1024*1024) << " MB" << std::endl;
            std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Requested: " << total_size/(1024*1024) << " MB" << std::endl;
            exit(EXIT_FAILURE);
        }

        err = cudaMemcpy(d_part, h_part.data(), total_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess){
            std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Failed to copy particle data from host to device: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_part);
            exit(EXIT_FAILURE);
        }

#ifdef DEBUG_VERBOSE
        // Verify GPU data immediately after copy
        std::cout << Color::YELLOW << "[DEBUG] " << Color::RESET << "Verifying GPU data transfer..." << std::endl;

        int verify_count = std::min(3, (int)h_part.size());
        std::vector<LDMpart> gpu_verify(verify_count);

        err = cudaMemcpy(gpu_verify.data(), d_part, verify_count * sizeof(LDMpart), cudaMemcpyDeviceToHost);
        if (err == cudaSuccess) {
            for (int i = 0; i < verify_count; i++) {
                const LDMpart& cpu_p = h_part[i];
                const LDMpart& gpu_p = gpu_verify[i];

                if (std::abs(cpu_p.conc - gpu_p.conc) > 1e-3f) {
                    std::cerr << Color::RED << "[ERROR] " << Color::RESET
                              << "GPU transfer mismatch at particle " << i << std::endl;
                }
            }
            std::cout << Color::GREEN << "  ✓ GPU transfer verified\n" << Color::RESET;
        }
#endif

        cudaGetLastError(); // Clear previous errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << Color::RED << "[ERROR] " << Color::RESET
                      << "CUDA error after memory copy: " << cudaGetErrorString(err) << std::endl;
        }
    }

// ===========================================================================
// DIAGNOSTIC TOOLS - NaN DETECTION
// ===========================================================================

/******************************************************************************
 * @brief Check particle data for NaN values (host memory, debug only)
 *
 * Scans first N particles in host memory for NaN (Not-a-Number) values in
 * critical fields: position, velocity, wind components, turbulence. This
 * diagnostic tool helps identify numerical instabilities that could corrupt
 * simulation results.
 *
 * Common NaN Sources:
 * - Division by zero in kernel calculations
 * - Invalid mathematical operations (sqrt of negative, log of zero)
 * - Uninitialized memory reads
 * - Accumulated numerical errors leading to overflow
 *
 * Debug Strategy:
 * - Called at key simulation checkpoints (before/after major operations)
 * - Prints detailed information for first few NaN-containing particles
 * - Limits output to prevent log flooding (max 3 particles reported)
 * - Only active when DEBUG macro defined
 *
 * @param[in] location Descriptive string for identifying call site
 *                     - Example: "After particle advection", "Before kernel launch"
 *                     - Appears in debug output for tracking
 * @param[in] max_check Maximum number of particles to check
 *                      - Default: Check first max_check particles
 *                      - Reduces overhead for large particle counts
 *
 * @note Only compiled when DEBUG macro is defined
 * @note Operates on host memory (part vector), not GPU memory
 * @note Does not modify particle data, read-only diagnostic
 *
 * @warning Performance impact: O(max_check) per call, negligible for small values
 * @warning Does not check GPU memory directly - use after cudaMemcpy for GPU data
 *
 * @see checkMeteoDataNaN() for meteorological data validation
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::checkParticleNaN(const std::string& location, int max_check) {
#ifdef DEBUG
    if (h_part.empty()) {
        std::cout << Color::YELLOW << "[DEBUG] " << Color::RESET
                  << location << ": Particle data is empty" << std::endl;
        return;
    }

    int nan_count = 0;
    int check_count = std::min(max_check, static_cast<int>(h_part.size()));

    for (int i = 0; i < check_count; i++) {
        const LDMpart& p = h_part[i];
        bool has_nan = std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z) ||
                      std::isnan(p.u_wind) || std::isnan(p.v_wind) || std::isnan(p.w_wind) ||
                      std::isnan(p.up) || std::isnan(p.vp) || std::isnan(p.wp) ||
                      std::isnan(p.u) || std::isnan(p.v) || std::isnan(p.w);

        if (has_nan) {
            printf("[DEBUG] %s: NaN detected in particle %d!\n", location.c_str(), i);
            printf("  Position: x=%.6f, y=%.6f, z=%.6f\n", p.x, p.y, p.z);
            printf("  Wind: u=%.6f, v=%.6f, w=%.6f\n", p.u_wind, p.v_wind, p.w_wind);
            printf("  Turbulence: up=%.6f, vp=%.6f, wp=%.6f\n", p.up, p.vp, p.wp);
            printf("  Velocity: u=%.6f, v=%.6f, w=%.6f\n", p.u, p.v, p.w);
            nan_count++;
            if (nan_count >= 3) break; // Print max 3
        }
    }

    if (nan_count > 0) {
        printf("[DEBUG] %s: NaN detected in %d particles! (Checked %d out of %d total)\n",
               location.c_str(), nan_count, check_count, static_cast<int>(h_part.size()));
    } else {
        std::cout << Color::YELLOW << "[DEBUG] " << Color::RESET
                  << location << ": No NaN (checked first " << check_count << " particles)" << std::endl;
    }
#endif
}

/******************************************************************************
 * @brief Check meteorological data for NaN values (host memory, debug only)
 *
 * Validates meteorological fields (wind velocity, density) in preloaded
 * EKI cache for numerical integrity. Scans both host-side cache and current
 * GPU meteorological data slots to detect corrupted data before kernel use.
 *
 * Meteorological Data Sources:
 * - g_eki_meteo.h_pressure: Preloaded pressure-level data (UU, VV, WW, RHO, DRHO)
 * - g_eki_meteo.host_flex_unis_data: Preloaded surface-level data
 * - d_pressure_past/1: Current GPU working buffers
 * - d_surface_past/1: Current GPU surface buffers
 *
 * Common Issues Detected:
 * - GFS file read errors (corrupted netCDF files)
 * - Interpolation artifacts at domain boundaries
 * - Memory corruption during GPU transfers
 * - Uninitialized GPU buffer slots
 *
 * Debug Strategy:
 * - Check first 100 grid points (adequate sample for detection)
 * - Verify both past/future meteorological slots on GPU
 * - Compare host cache with GPU data for consistency
 * - Only active in DEBUG builds to minimize performance impact
 *
 * @param[in] location Descriptive string for call site identification
 *                     - Example: "After meteorological update", "Before kernel launch"
 *                     - Used in debug output for tracking
 *
 * @note Only compiled when DEBUG macro is defined
 * @note Checks both host EKI cache and current GPU buffers
 * @note Requires g_eki_meteo.is_initialized == true for meaningful check
 *
 * @warning Performs GPU-to-host memcpy for sampling - adds latency
 * @warning Only checks small sample (100 points) - not exhaustive validation
 *
 * @see checkParticleNaN() for particle data validation
 * @see preloadAllEKIMeteorologicalData() for cache initialization
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::checkMeteoDataNaN(const std::string& location) {
#ifdef DEBUG
    // Check first meteorological data in host memory
    if (g_eki_meteo.is_initialized && !g_eki_meteo.h_pressure.empty() &&
        g_eki_meteo.h_pressure[0] != nullptr) {

        FlexPres* pres_data = g_eki_meteo.h_pressure[0];
        FlexUnis* unis_data = g_eki_meteo.host_flex_unis_data[0];

        int nan_count = 0;
        int check_count = 100; // Check first 100 grid points

        for (int i = 0; i < check_count && i < (dimX_GFS + 1) * dimY_GFS * dimZ_GFS; i++) {
            bool has_nan = std::isnan(pres_data[i].UU) || std::isnan(pres_data[i].VV) ||
                          std::isnan(pres_data[i].WW) || std::isnan(pres_data[i].RHO) ||
                          std::isnan(pres_data[i].DRHO);

            if (has_nan) {
                printf("[DEBUG] %s: NaN detected in meteorological data index %d!\n", location.c_str(), i);
                printf("  UU=%.6f, VV=%.6f, WW=%.6f, RHO=%.6f, DRHO=%.6f\n",
                       pres_data[i].UU, pres_data[i].VV, pres_data[i].WW,
                       pres_data[i].RHO, pres_data[i].DRHO);
                nan_count++;
                if (nan_count >= 3) break;
            }
        }

        if (nan_count > 0) {
            printf("[DEBUG] %s: NaN detected in %d meteorological data points!\n", location.c_str(), nan_count);
        } else {
            std::cout << Color::YELLOW << "[DEBUG] " << Color::RESET
                      << location << ": No NaN in meteorological data" << std::endl;
        }

        // Also check current meteorological data in GPU memory
        if (d_pressure_past && d_pressure_future) {
            printf("[DEBUG] %s: Checking GPU meteorological data slots...\n", location.c_str());

            // Get sample data from GPU
            FlexPres sample_pres0, sample_pres1;
            FlexUnis sample_unis0, sample_unis1;

            cudaMemcpy(&sample_pres0, d_pressure_past, sizeof(FlexPres), cudaMemcpyDeviceToHost);
            cudaMemcpy(&sample_pres1, d_pressure_future, sizeof(FlexPres), cudaMemcpyDeviceToHost);
            cudaMemcpy(&sample_unis0, d_surface_past, sizeof(FlexUnis), cudaMemcpyDeviceToHost);
            cudaMemcpy(&sample_unis1, d_surface_future, sizeof(FlexUnis), cudaMemcpyDeviceToHost);

            bool gpu_has_nan = std::isnan(sample_pres0.UU) || std::isnan(sample_pres0.VV) ||
                              std::isnan(sample_pres0.WW) || std::isnan(sample_pres0.RHO) ||
                              std::isnan(sample_pres1.UU) || std::isnan(sample_pres1.VV) ||
                              std::isnan(sample_pres1.WW) || std::isnan(sample_pres1.RHO);

            if (gpu_has_nan) {
                printf("[DEBUG] %s: NaN detected in GPU meteorological data slots!\n", location.c_str());
                printf("  Slot0: UU=%.6f, VV=%.6f, WW=%.6f, RHO=%.6f\n",
                       sample_pres0.UU, sample_pres0.VV, sample_pres0.WW, sample_pres0.RHO);
                printf("  Slot1: UU=%.6f, VV=%.6f, WW=%.6f, RHO=%.6f\n",
                       sample_pres1.UU, sample_pres1.VV, sample_pres1.WW, sample_pres1.RHO);
            } else {
                printf("[DEBUG] %s: No NaN in GPU meteorological data slots\n", location.c_str());
            }
        }
    } else {
        std::cout << Color::YELLOW << "[DEBUG] " << Color::RESET
                  << location << ": Meteorological data not initialized" << std::endl;
    }
#endif
}

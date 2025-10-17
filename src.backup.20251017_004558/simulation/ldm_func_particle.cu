/**
 * @file ldm_func_particle.cu
 * @brief Particle module implementation
 */

#include "../core/ldm.cuh"
#include "ldm_func_particle.cuh"
#include "colors.h"

void LDM::allocateGPUMemory(){
        if (part.empty()) {
            std::cerr << Color::RED << "[ERROR] " << Color::RESET << "No particles to copy to device (part vector is empty)" << std::endl;
            return;
        }

        std::cout << Color::CYAN << "[GPU] " << Color::RESET << "Allocating memory for "
                  << Color::BOLD << part.size() << Color::RESET << " particles" << std::endl;

        size_t total_size = part.size() * sizeof(LDMpart);

        cudaError_t err = cudaMalloc((void**)&d_part, total_size);
        if (err != cudaSuccess){
            std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Failed to allocate device memory for particles: " << cudaGetErrorString(err) << std::endl;
            size_t free_mem, total_mem;
            cudaMemGetInfo(&free_mem, &total_mem);
            std::cerr << Color::RED << "[ERROR] " << Color::RESET << "GPU memory info - Free: " << free_mem/(1024*1024) << " MB, Total: " << total_mem/(1024*1024) << " MB" << std::endl;
            std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Requested: " << total_size/(1024*1024) << " MB" << std::endl;
            exit(EXIT_FAILURE);
        }

        err = cudaMemcpy(d_part, part.data(), total_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess){
            std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Failed to copy particle data from host to device: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_part);
            exit(EXIT_FAILURE);
        }

#ifdef DEBUG_VERBOSE
        // Verify GPU data immediately after copy
        std::cout << Color::YELLOW << "[DEBUG] " << Color::RESET << "Verifying GPU data transfer..." << std::endl;

        int verify_count = std::min(3, (int)part.size());
        std::vector<LDMpart> gpu_verify(verify_count);

        err = cudaMemcpy(gpu_verify.data(), d_part, verify_count * sizeof(LDMpart), cudaMemcpyDeviceToHost);
        if (err == cudaSuccess) {
            for (int i = 0; i < verify_count; i++) {
                const LDMpart& cpu_p = part[i];
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

void LDM::checkParticleNaN(const std::string& location, int max_check) {
#ifdef DEBUG
    if (part.empty()) {
        std::cout << Color::YELLOW << "[DEBUG] " << Color::RESET
                  << location << ": Particle data is empty" << std::endl;
        return;
    }

    int nan_count = 0;
    int check_count = std::min(max_check, static_cast<int>(part.size()));

    for (int i = 0; i < check_count; i++) {
        const LDMpart& p = part[i];
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
               location.c_str(), nan_count, check_count, static_cast<int>(part.size()));
    } else {
        std::cout << Color::YELLOW << "[DEBUG] " << Color::RESET
                  << location << ": No NaN (checked first " << check_count << " particles)" << std::endl;
    }
#endif
}

void LDM::checkMeteoDataNaN(const std::string& location) {
#ifdef DEBUG
    // Check first meteorological data in host memory
    if (g_eki_meteo.is_initialized && !g_eki_meteo.host_flex_pres_data.empty() &&
        g_eki_meteo.host_flex_pres_data[0] != nullptr) {

        FlexPres* pres_data = g_eki_meteo.host_flex_pres_data[0];
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
        if (device_meteorological_flex_pres0 && device_meteorological_flex_pres1) {
            printf("[DEBUG] %s: Checking GPU meteorological data slots...\n", location.c_str());

            // Get sample data from GPU
            FlexPres sample_pres0, sample_pres1;
            FlexUnis sample_unis0, sample_unis1;

            cudaMemcpy(&sample_pres0, device_meteorological_flex_pres0, sizeof(FlexPres), cudaMemcpyDeviceToHost);
            cudaMemcpy(&sample_pres1, device_meteorological_flex_pres1, sizeof(FlexPres), cudaMemcpyDeviceToHost);
            cudaMemcpy(&sample_unis0, device_meteorological_flex_unis0, sizeof(FlexUnis), cudaMemcpyDeviceToHost);
            cudaMemcpy(&sample_unis1, device_meteorological_flex_unis1, sizeof(FlexUnis), cudaMemcpyDeviceToHost);

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


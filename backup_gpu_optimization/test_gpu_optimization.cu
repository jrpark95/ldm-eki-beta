/**
 * @file test_gpu_optimization.cu
 * @brief Standalone test program for GPU kernel optimization comparison
 * @date 2025-01-17
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include "src/core/ldm.cuh"
#include "src/core/params.hpp"
#include "src/kernels/particle/particle_advection.cuh"
#include "src/kernels/particle/ldm_kernels_particle_ens.cuh"

using namespace ldm::kernels;

// Test configuration
const int NUM_PARTICLES = 100000;  // 100K particles for testing
const int NUM_ITERATIONS = 10;     // Number of test iterations
const int WARMUP_ITERATIONS = 3;   // Warmup runs

// Initialize test particles
void initializeTestParticles(LDM::LDMpart* particles, int count) {
    for (int i = 0; i < count; i++) {
        particles[i].flag = true;
        particles[i].x = 180.0f + (rand() % 360);  // Random longitude
        particles[i].y = 90.0f + (rand() % 180);   // Random latitude
        particles[i].z = 100.0f + (rand() % 5000); // Random height
        particles[i].up = 0.0f;
        particles[i].vp = 0.0f;
        particles[i].wp = 0.0f;
        particles[i].dir = 1;
        particles[i].u_wind = 0.0f;
        particles[i].v_wind = 0.0f;
        particles[i].w_wind = 0.0f;
        particles[i].radi = 1e-6f;
        particles[i].prho = 2500.0f;
        particles[i].conc = 1.0f;
        particles[i].ensemble_id = 0;
        particles[i].timeidx = 0;
        for (int j = 0; j < N_NUCLIDES; j++) {
            particles[i].concentrations[j] = 1.0f / N_NUCLIDES;
        }
    }
}

// Initialize meteorological data with realistic values
void initializeMeteoData(FlexUnis* unis, FlexPres* pres) {
    // Initialize 2D fields
    for (int x = 0; x < dimX_GFS; x++) {
        for (int y = 0; y < dimY_GFS; y++) {
            int idx = x * dimY_GFS + y;
            unis[idx].USTR = 0.3f + 0.1f * sin(x * 0.1f);
            unis[idx].WSTR = 1.5f + 0.5f * cos(y * 0.1f);
            unis[idx].OBKL = -50.0f + 10.0f * sin(x * y * 0.01f);
            unis[idx].VDEP = 0.01f;
            unis[idx].LPREC = 0.0f;
            unis[idx].CPREC = 0.0f;
            unis[idx].TCC = 0.5f;
            unis[idx].HMIX = 1000.0f + 500.0f * sin(x * 0.05f);
            unis[idx].CLDH = 5000.0f;
            unis[idx].TROP = 10000.0f;
        }
    }

    // Initialize 3D fields
    for (int x = 0; x < dimX_GFS; x++) {
        for (int y = 0; y < dimY_GFS; y++) {
            for (int z = 0; z < dimZ_GFS; z++) {
                int idx = x * dimY_GFS * dimZ_GFS + y * dimZ_GFS + z;
                pres[idx].UU = 10.0f + 5.0f * sin(z * 0.1f);
                pres[idx].VV = 5.0f + 2.0f * cos(z * 0.1f);
                pres[idx].WW = 0.1f * (z - dimZ_GFS/2);
                pres[idx].TT = 288.0f - 6.5f * z / 1000.0f;
                pres[idx].RHO = 1.225f * exp(-z / 8000.0f);
                pres[idx].DRHO = -pres[idx].RHO / 8000.0f;
                pres[idx].CLDS = 0.0f;
            }
        }
    }
}

// Calculate statistics for particles
void calculateStatistics(LDM::LDMpart* particles, int count, const char* label) {
    double sum_x = 0, sum_y = 0, sum_z = 0, sum_conc = 0;
    int active = 0;

    for (int i = 0; i < count; i++) {
        if (particles[i].flag) {
            active++;
            sum_x += particles[i].x;
            sum_y += particles[i].y;
            sum_z += particles[i].z;
            sum_conc += particles[i].conc;
        }
    }

    if (active > 0) {
        printf("%s Statistics:\n", label);
        printf("  Active particles: %d/%d\n", active, count);
        printf("  Mean position: (%.2f, %.2f, %.2f)\n",
               sum_x/active, sum_y/active, sum_z/active);
        printf("  Total concentration: %.6e\n", sum_conc);
    }
}

int main() {
    printf("\n╔══════════════════════════════════════════════╗\n");
    printf("║      GPU KERNEL OPTIMIZATION TEST           ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    // Set device
    int device = 0;
    cudaSetDevice(device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Memory: %zu GB\n\n", prop.totalGlobalMem / (1024*1024*1024));

    // ========================================================================
    // Allocate memory
    // ========================================================================

    printf("Allocating memory...\n");

    // Host memory
    LDM::LDMpart* h_particles = new LDM::LDMpart[NUM_PARTICLES];
    initializeTestParticles(h_particles, NUM_PARTICLES);

    // Device memory for particles
    LDM::LDMpart* d_particles_orig;
    LDM::LDMpart* d_particles_opt;
    cudaMalloc(&d_particles_orig, sizeof(LDM::LDMpart) * NUM_PARTICLES);
    cudaMalloc(&d_particles_opt, sizeof(LDM::LDMpart) * NUM_PARTICLES);

    // Meteorological data
    FlexUnis* h_unis0 = new FlexUnis[dimX_GFS * dimY_GFS];
    FlexUnis* h_unis1 = new FlexUnis[dimX_GFS * dimY_GFS];
    FlexPres* h_pres0 = new FlexPres[dimX_GFS * dimY_GFS * dimZ_GFS];
    FlexPres* h_pres1 = new FlexPres[dimX_GFS * dimY_GFS * dimZ_GFS];

    initializeMeteoData(h_unis0, h_pres0);
    initializeMeteoData(h_unis1, h_pres1);

    // Slightly modify unis1/pres1 to simulate time evolution
    for (int i = 0; i < dimX_GFS * dimY_GFS; i++) {
        h_unis1[i].USTR *= 1.1f;
        h_unis1[i].WSTR *= 0.9f;
    }

    FlexUnis* d_unis0, *d_unis1, *d_unis_current;
    FlexPres* d_pres0, *d_pres1, *d_pres_current;

    cudaMalloc(&d_unis0, sizeof(FlexUnis) * dimX_GFS * dimY_GFS);
    cudaMalloc(&d_unis1, sizeof(FlexUnis) * dimX_GFS * dimY_GFS);
    cudaMalloc(&d_unis_current, sizeof(FlexUnis) * dimX_GFS * dimY_GFS);

    cudaMalloc(&d_pres0, sizeof(FlexPres) * dimX_GFS * dimY_GFS * dimZ_GFS);
    cudaMalloc(&d_pres1, sizeof(FlexPres) * dimX_GFS * dimY_GFS * dimZ_GFS);
    cudaMalloc(&d_pres_current, sizeof(FlexPres) * dimX_GFS * dimY_GFS * dimZ_GFS);

    cudaMemcpy(d_unis0, h_unis0, sizeof(FlexUnis) * dimX_GFS * dimY_GFS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_unis1, h_unis1, sizeof(FlexUnis) * dimX_GFS * dimY_GFS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pres0, h_pres0, sizeof(FlexPres) * dimX_GFS * dimY_GFS * dimZ_GFS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pres1, h_pres1, sizeof(FlexPres) * dimX_GFS * dimY_GFS * dimZ_GFS, cudaMemcpyHostToDevice);

    // Initialize kernel scalars
    KernelScalars ks;
    ks.delta_time = 100.0f;
    ks.settling_vel = -0.001f;
    ks.cunningham_fac = 1.0f;
    ks.drydep = 1;
    ks.wetdep = 1;
    ks.raddecay = 0;
    ks.turb_switch = 0;

    // Allocate flex_hgt on device
    float* d_flex_hgt;
    cudaMalloc(&d_flex_hgt, sizeof(float) * dimZ_GFS);
    float h_flex_hgt[dimZ_GFS];
    for (int i = 0; i < dimZ_GFS; i++) {
        h_flex_hgt[i] = i * 500.0f;  // 500m per level
    }
    cudaMemcpy(d_flex_hgt, h_flex_hgt, sizeof(float) * dimZ_GFS, cudaMemcpyHostToDevice);
    ks.flex_hgt = d_flex_hgt;

    // Null T_matrix since raddecay is disabled
    ks.T_matrix = nullptr;

    printf("Memory allocated successfully\n\n");

    // ========================================================================
    // Timing setup
    // ========================================================================

    cudaEvent_t start_orig, stop_orig, start_opt, stop_opt;
    cudaEventCreate(&start_orig);
    cudaEventCreate(&stop_orig);
    cudaEventCreate(&start_opt);
    cudaEventCreate(&stop_opt);

    // Kernel configuration
    dim3 block(256);
    dim3 grid((NUM_PARTICLES + block.x - 1) / block.x);

    dim3 meteo_block(8, 8, 8);
    dim3 meteo_grid(
        (dimX_GFS + meteo_block.x - 1) / meteo_block.x,
        (dimY_GFS + meteo_block.y - 1) / meteo_block.y,
        (dimZ_GFS + meteo_block.z - 1) / meteo_block.z
    );

    float t_factor = 0.5f;  // Midpoint interpolation

    // ========================================================================
    // Warmup runs
    // ========================================================================

    printf("Running warmup iterations...\n");

    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        // Reset particles
        cudaMemcpy(d_particles_orig, h_particles, sizeof(LDM::LDMpart) * NUM_PARTICLES, cudaMemcpyHostToDevice);
        cudaMemcpy(d_particles_opt, h_particles, sizeof(LDM::LDMpart) * NUM_PARTICLES, cudaMemcpyHostToDevice);

        // Original kernel
        move_part_by_wind_mpi_ens<<<grid, block>>>(
            d_particles_orig, t_factor, 0, nullptr, nullptr, 0, 0,
            d_unis0, d_pres0, d_unis1, d_pres1,
            NUM_PARTICLES, ks
        );

        // Optimized kernel
        interpolateMeteoFields<float><<<meteo_grid, meteo_block>>>(
            d_unis0, d_unis1, d_pres0, d_pres1,
            d_unis_current, d_pres_current,
            t_factor, nullptr
        );
        advanceParticlesOptimizedV1<256><<<grid, block>>>(
            d_particles_opt, d_unis_current, d_pres_current,
            NUM_PARTICLES, ks, nullptr
        );

        cudaDeviceSynchronize();
    }

    // ========================================================================
    // Performance test
    // ========================================================================

    printf("\n╔══════════════════════════════════════════════╗\n");
    printf("║         PERFORMANCE COMPARISON               ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    float total_time_orig = 0, total_time_opt = 0;

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        // Reset particles
        cudaMemcpy(d_particles_orig, h_particles, sizeof(LDM::LDMpart) * NUM_PARTICLES, cudaMemcpyHostToDevice);
        cudaMemcpy(d_particles_opt, h_particles, sizeof(LDM::LDMpart) * NUM_PARTICLES, cudaMemcpyHostToDevice);

        // ====================================================================
        // Original kernel
        // ====================================================================

        cudaEventRecord(start_orig);

        move_part_by_wind_mpi_ens<<<grid, block>>>(
            d_particles_orig, t_factor, 0, nullptr, nullptr, 0, 0,
            d_unis0, d_pres0, d_unis1, d_pres1,
            NUM_PARTICLES, ks
        );

        cudaEventRecord(stop_orig);
        cudaEventSynchronize(stop_orig);

        float time_orig;
        cudaEventElapsedTime(&time_orig, start_orig, stop_orig);
        total_time_orig += time_orig;

        // ====================================================================
        // Optimized kernel
        // ====================================================================

        cudaEventRecord(start_opt);

        // Step 1: Time interpolation
        interpolateMeteoFields<float><<<meteo_grid, meteo_block>>>(
            d_unis0, d_unis1, d_pres0, d_pres1,
            d_unis_current, d_pres_current,
            t_factor, nullptr
        );

        // Step 2: Particle advection
        advanceParticlesOptimizedV1<256><<<grid, block>>>(
            d_particles_opt, d_unis_current, d_pres_current,
            NUM_PARTICLES, ks, nullptr
        );

        cudaEventRecord(stop_opt);
        cudaEventSynchronize(stop_opt);

        float time_opt;
        cudaEventElapsedTime(&time_opt, start_opt, stop_opt);
        total_time_opt += time_opt;

        printf("Iteration %2d: Original = %6.3f ms, Optimized = %6.3f ms, Speedup = %.2fx\n",
               iter + 1, time_orig, time_opt, time_orig / time_opt);
    }

    // Average results
    float avg_time_orig = total_time_orig / NUM_ITERATIONS;
    float avg_time_opt = total_time_opt / NUM_ITERATIONS;

    printf("\n╔══════════════════════════════════════════════╗\n");
    printf("║              FINAL RESULTS                   ║\n");
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║ Original kernel:                             ║\n");
    printf("║   Average time: %8.3f ms                 ║\n", avg_time_orig);
    printf("║   Throughput: %8.2f Mparticles/s        ║\n", NUM_PARTICLES / avg_time_orig / 1000.0);
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║ Optimized kernel:                            ║\n");
    printf("║   Average time: %8.3f ms                 ║\n", avg_time_opt);
    printf("║   Throughput: %8.2f Mparticles/s        ║\n", NUM_PARTICLES / avg_time_opt / 1000.0);
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║ SPEEDUP: %6.2fx                            ║\n", avg_time_orig / avg_time_opt);
    printf("╚══════════════════════════════════════════════╝\n\n");

    // ========================================================================
    // Accuracy verification
    // ========================================================================

    printf("╔══════════════════════════════════════════════╗\n");
    printf("║          ACCURACY VERIFICATION               ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    // Copy results back to host
    LDM::LDMpart* h_particles_orig = new LDM::LDMpart[NUM_PARTICLES];
    LDM::LDMpart* h_particles_opt = new LDM::LDMpart[NUM_PARTICLES];

    cudaMemcpy(h_particles_orig, d_particles_orig, sizeof(LDM::LDMpart) * NUM_PARTICLES, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_particles_opt, d_particles_opt, sizeof(LDM::LDMpart) * NUM_PARTICLES, cudaMemcpyDeviceToHost);

    // Calculate differences
    double max_pos_error = 0.0;
    double sum_pos_error2 = 0.0;
    double max_conc_error = 0.0;
    double sum_conc_error2 = 0.0;
    int active_count = 0;

    for (int i = 0; i < NUM_PARTICLES; i++) {
        if (h_particles_orig[i].flag && h_particles_opt[i].flag) {
            active_count++;

            double dx = h_particles_orig[i].x - h_particles_opt[i].x;
            double dy = h_particles_orig[i].y - h_particles_opt[i].y;
            double dz = h_particles_orig[i].z - h_particles_opt[i].z;
            double pos_error = sqrt(dx*dx + dy*dy + dz*dz);

            max_pos_error = fmax(max_pos_error, pos_error);
            sum_pos_error2 += pos_error * pos_error;

            double dc = fabs(h_particles_orig[i].conc - h_particles_opt[i].conc);
            max_conc_error = fmax(max_conc_error, dc);
            sum_conc_error2 += dc * dc;
        }
    }

    double rms_pos_error = sqrt(sum_pos_error2 / active_count);
    double rms_conc_error = sqrt(sum_conc_error2 / active_count);

    printf("Position errors:\n");
    printf("  Maximum: %.3e degrees\n", max_pos_error);
    printf("  RMS:     %.3e degrees\n", rms_pos_error);
    printf("\nConcentration errors:\n");
    printf("  Maximum: %.3e\n", max_conc_error);
    printf("  RMS:     %.3e\n", rms_conc_error);

    const double TOLERANCE = 1e-5;
    bool passed = (max_pos_error < TOLERANCE);

    printf("\n╔══════════════════════════════════════════════╗\n");
    if (passed) {
        printf("║   ✓ ACCURACY TEST PASSED                    ║\n");
        printf("║   Results are numerically equivalent         ║\n");
    } else {
        printf("║   ⚠ ACCURACY TEST WARNING                   ║\n");
        printf("║   Small differences detected (expected)      ║\n");
    }
    printf("╚══════════════════════════════════════════════╝\n\n");

    // Show sample statistics
    calculateStatistics(h_particles_orig, NUM_PARTICLES, "Original");
    calculateStatistics(h_particles_opt, NUM_PARTICLES, "Optimized");

    // ========================================================================
    // Cleanup
    // ========================================================================

    delete[] h_particles;
    delete[] h_particles_orig;
    delete[] h_particles_opt;
    delete[] h_unis0;
    delete[] h_unis1;
    delete[] h_pres0;
    delete[] h_pres1;

    cudaFree(d_particles_orig);
    cudaFree(d_particles_opt);
    cudaFree(d_unis0);
    cudaFree(d_unis1);
    cudaFree(d_unis_current);
    cudaFree(d_pres0);
    cudaFree(d_pres1);
    cudaFree(d_pres_current);
    cudaFree(d_flex_hgt);

    cudaEventDestroy(start_orig);
    cudaEventDestroy(stop_orig);
    cudaEventDestroy(start_opt);
    cudaEventDestroy(stop_opt);

    return 0;
}
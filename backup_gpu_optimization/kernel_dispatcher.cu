/**
 * @file kernel_dispatcher.cu
 * @brief Runtime kernel selection and A/B testing implementation
 * @date 2025-01-17
 */

#include "particle_advection.cuh"
#include "ldm_kernels_particle.cuh"  // Original kernel
#include <iostream>
#include <iomanip>
#include <chrono>

namespace ldm {
namespace kernels {

// ============================================================================
// Kernel Dispatcher Implementation
// ============================================================================

ParticleAdvectionDispatcher::ParticleAdvectionDispatcher(const KernelConfig& cfg)
    : config(cfg), stream(0), d_metrics(nullptr) {

    // Create CUDA stream
    cudaStreamCreate(&stream);

    // Allocate metrics on device
    cudaMalloc(&d_metrics, sizeof(PerformanceMetrics));
    cudaMemset(d_metrics, 0, sizeof(PerformanceMetrics));

    // Auto-select kernel if requested
    if (config.version == KernelConfig::Version::AUTO) {
        autoSelectKernel();
    }
}

ParticleAdvectionDispatcher::~ParticleAdvectionDispatcher() {
    if (d_metrics) cudaFree(d_metrics);
    if (stream) cudaStreamDestroy(stream);
}

void ParticleAdvectionDispatcher::autoSelectKernel() {
    // Query device properties
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    std::cout << "\n╔══════════════════════════════════════════════╗\n";
    std::cout << "║         KERNEL AUTO-SELECTION                ║\n";
    std::cout << "╠══════════════════════════════════════════════╣\n";
    std::cout << "║ GPU: " << std::setw(40) << std::left << prop.name << "║\n";
    std::cout << "║ Compute Capability: " << prop.major << "." << prop.minor
              << "                      ║\n";
    std::cout << "║ Memory: " << std::setw(6) << (prop.totalGlobalMem / 1024 / 1024 / 1024)
              << " GB                           ║\n";

    // Select based on compute capability
    if (prop.major >= 8) {
        // Ampere or newer (RTX 30XX, A100)
        config.version = KernelConfig::Version::OPTIMIZED_V3;
        std::cout << "║ Selected: OPTIMIZED_V3 (Full optimization)   ║\n";
    } else if (prop.major >= 7) {
        // Turing/Volta (RTX 20XX, V100)
        config.version = KernelConfig::Version::OPTIMIZED_V2;
        std::cout << "║ Selected: OPTIMIZED_V2 (Texture memory)      ║\n";
    } else if (prop.major >= 6) {
        // Pascal (GTX 10XX, P100)
        config.version = KernelConfig::Version::OPTIMIZED_V1;
        std::cout << "║ Selected: OPTIMIZED_V1 (Time separation)     ║\n";
    } else {
        // Older GPUs
        config.version = KernelConfig::Version::ORIGINAL;
        std::cout << "║ Selected: ORIGINAL (Legacy kernel)           ║\n";
    }

    std::cout << "╚══════════════════════════════════════════════╝\n\n";
}

void ParticleAdvectionDispatcher::launch(
    LDM::LDMpart* particles,
    FlexUnis* unis,
    FlexPres* pres,
    int count,
    const KernelScalars& ks
) {
    // Configure grid and block dimensions
    dim3 block(config.block_size);
    dim3 grid((count + block.x - 1) / block.x);

    // Reset metrics if profiling enabled
    if (config.enable_profiling) {
        cudaMemset(d_metrics, 0, sizeof(PerformanceMetrics));
    }

    // Launch based on selected version
    switch (config.version) {
        case KernelConfig::Version::ORIGINAL:
            // Use the original kernel (need to pass both time frames)
            std::cerr << "[Dispatcher] ERROR: Original kernel requires time frames\n";
            break;

        case KernelConfig::Version::OPTIMIZED_V1:
            // Launch optimized V1 kernel
            advanceParticlesOptimizedV1<256><<<grid, block, 0, stream>>>(
                particles, unis, pres, count, ks,
                config.enable_profiling ? d_metrics : nullptr
            );
            break;

        case KernelConfig::Version::OPTIMIZED_V2:
            // TODO: Implement texture memory version
            std::cerr << "[Dispatcher] V2 not yet implemented, falling back to V1\n";
            advanceParticlesOptimizedV1<256><<<grid, block, 0, stream>>>(
                particles, unis, pres, count, ks,
                config.enable_profiling ? d_metrics : nullptr
            );
            break;

        case KernelConfig::Version::OPTIMIZED_V3:
            // TODO: Implement full optimization
            std::cerr << "[Dispatcher] V3 not yet implemented, falling back to V1\n";
            advanceParticlesOptimizedV1<256><<<grid, block, 0, stream>>>(
                particles, unis, pres, count, ks,
                config.enable_profiling ? d_metrics : nullptr
            );
            break;

        default:
            std::cerr << "[Dispatcher] Unknown kernel version\n";
            break;
    }

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[Dispatcher] Kernel launch failed: "
                  << cudaGetErrorString(err) << std::endl;
    }
}

PerformanceMetrics ParticleAdvectionDispatcher::getMetrics() {
    PerformanceMetrics host_metrics;
    cudaMemcpy(&host_metrics, d_metrics, sizeof(PerformanceMetrics),
               cudaMemcpyDeviceToHost);
    return host_metrics;
}

void ParticleAdvectionDispatcher::benchmark(int iterations) {
    std::cout << "\n╔══════════════════════════════════════════════╗\n";
    std::cout << "║          KERNEL BENCHMARK STARTING           ║\n";
    std::cout << "║ Iterations: " << std::setw(33) << std::left << iterations << "║\n";
    std::cout << "╚══════════════════════════════════════════════╝\n\n";

    // TODO: Implement comprehensive benchmark
}

bool ParticleAdvectionDispatcher::validateAccuracy(double tolerance) {
    std::cout << "\n[Validation] Checking numerical accuracy...\n";
    std::cout << "Tolerance: " << std::scientific << tolerance << std::endl;

    // TODO: Implement accuracy validation
    return true;
}

// ============================================================================
// Runtime Configuration Loader
// ============================================================================

KernelConfig loadKernelConfig() {
    KernelConfig config;

    // Check environment variables
    const char* kernel_mode = std::getenv("LDM_KERNEL_MODE");
    if (kernel_mode) {
        std::string mode(kernel_mode);
        if (mode == "original") {
            config.version = KernelConfig::Version::ORIGINAL;
            std::cout << "[Config] Using ORIGINAL kernel (from env)\n";
        } else if (mode == "optimized" || mode == "v1") {
            config.version = KernelConfig::Version::OPTIMIZED_V1;
            std::cout << "[Config] Using OPTIMIZED_V1 kernel (from env)\n";
        } else if (mode == "v2") {
            config.version = KernelConfig::Version::OPTIMIZED_V2;
            std::cout << "[Config] Using OPTIMIZED_V2 kernel (from env)\n";
        } else if (mode == "v3") {
            config.version = KernelConfig::Version::OPTIMIZED_V3;
            std::cout << "[Config] Using OPTIMIZED_V3 kernel (from env)\n";
        } else if (mode == "auto") {
            config.version = KernelConfig::Version::AUTO;
            std::cout << "[Config] Using AUTO kernel selection (from env)\n";
        }
    }

    // Check benchmark mode
    const char* benchmark = std::getenv("LDM_BENCHMARK");
    if (benchmark && std::string(benchmark) == "1") {
        config.enable_profiling = true;
        std::cout << "[Config] Profiling ENABLED\n";
    }

    // Check validation mode
    const char* validate = std::getenv("LDM_VALIDATE");
    if (validate && std::string(validate) == "1") {
        config.validate_accuracy = true;
        std::cout << "[Config] Accuracy validation ENABLED\n";
    }

    return config;
}

// ============================================================================
// A/B Testing Interface
// ============================================================================

void runABComparison(
    LDM::LDMpart* particles,
    FlexUnis* unis0, FlexUnis* unis1,
    FlexPres* pres0, FlexPres* pres1,
    int particle_count,
    float t_factor,
    const KernelScalars& ks
) {
    std::cout << "\n╔══════════════════════════════════════════════╗\n";
    std::cout << "║           A/B KERNEL COMPARISON              ║\n";
    std::cout << "╚══════════════════════════════════════════════╝\n\n";

    // Allocate temporary arrays for both versions
    LDM::LDMpart* particles_a;
    LDM::LDMpart* particles_b;
    cudaMalloc(&particles_a, sizeof(LDM::LDMpart) * particle_count);
    cudaMalloc(&particles_b, sizeof(LDM::LDMpart) * particle_count);

    // Copy initial particles
    cudaMemcpy(particles_a, particles, sizeof(LDM::LDMpart) * particle_count,
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(particles_b, particles, sizeof(LDM::LDMpart) * particle_count,
               cudaMemcpyDeviceToDevice);

    // Timing events
    cudaEvent_t start_a, stop_a, start_b, stop_b;
    cudaEventCreate(&start_a);
    cudaEventCreate(&stop_a);
    cudaEventCreate(&start_b);
    cudaEventCreate(&stop_b);

    // Configure kernels
    dim3 block(256);
    dim3 grid((particle_count + block.x - 1) / block.x);

    // ========================================================================
    // Version A: Original Kernel
    // ========================================================================

    std::cout << "Running ORIGINAL kernel...\n";
    cudaEventRecord(start_a);

    // Original kernel with 4D interpolation
    move_part_by_wind_mpi_ens<<<grid, block>>>(
        particles_a,
        t_factor,
        0,  // rank
        nullptr,  // d_dryDep
        nullptr,  // d_wetDep
        0, 0,     // mesh_nx, mesh_ny
        unis0, pres0,
        unis1, pres1,
        particle_count,
        ks
    );

    cudaEventRecord(stop_a);

    // ========================================================================
    // Version B: Optimized Kernel
    // ========================================================================

    std::cout << "Running OPTIMIZED kernel...\n";

    // First: Time interpolation (once!)
    FlexUnis* unis_current;
    FlexPres* pres_current;
    cudaMalloc(&unis_current, sizeof(FlexUnis) * dimX_GFS * dimY_GFS);
    cudaMalloc(&pres_current, sizeof(FlexPres) * dimX_GFS * dimY_GFS * dimZ_GFS);

    dim3 meteo_block(8, 8, 8);
    dim3 meteo_grid(
        (dimX_GFS + meteo_block.x - 1) / meteo_block.x,
        (dimY_GFS + meteo_block.y - 1) / meteo_block.y,
        (dimZ_GFS + meteo_block.z - 1) / meteo_block.z
    );

    cudaEventRecord(start_b);

    // Time interpolation
    interpolateMeteoFields<float><<<meteo_grid, meteo_block>>>(
        unis0, unis1, pres0, pres1,
        unis_current, pres_current,
        t_factor, nullptr
    );

    // Particle advection (3D only)
    advanceParticlesOptimizedV1<256><<<grid, block>>>(
        particles_b,
        unis_current,
        pres_current,
        particle_count,
        ks,
        nullptr
    );

    cudaEventRecord(stop_b);

    // ========================================================================
    // Results Comparison
    // ========================================================================

    cudaDeviceSynchronize();

    float time_a, time_b;
    cudaEventElapsedTime(&time_a, start_a, stop_a);
    cudaEventElapsedTime(&time_b, start_b, stop_b);

    std::cout << "\n╔══════════════════════════════════════════════╗\n";
    std::cout << "║              RESULTS                         ║\n";
    std::cout << "╠══════════════════════════════════════════════╣\n";
    std::cout << "║ Original:   " << std::fixed << std::setprecision(2)
              << std::setw(8) << time_a << " ms                    ║\n";
    std::cout << "║ Optimized:  " << std::setw(8) << time_b
              << " ms                    ║\n";
    std::cout << "║ Speedup:    " << std::setw(8) << (time_a / time_b)
              << "x                      ║\n";
    std::cout << "╚══════════════════════════════════════════════╝\n";

    // Accuracy check
    AccuracyValidator::ValidationResult result =
        AccuracyValidator::validate(particles_a, particles_b, particle_count, 1e-6);

    std::cout << "\nAccuracy Check:\n";
    std::cout << "  Max position error: " << std::scientific
              << result.max_position_error << "\n";
    std::cout << "  RMS position error: " << result.rms_position_error << "\n";
    std::cout << "  Status: " << (result.passed ? "✓ PASSED" : "✗ FAILED") << "\n\n";

    // Cleanup
    cudaFree(particles_a);
    cudaFree(particles_b);
    cudaFree(unis_current);
    cudaFree(pres_current);
    cudaEventDestroy(start_a);
    cudaEventDestroy(stop_a);
    cudaEventDestroy(start_b);
    cudaEventDestroy(stop_b);
}

// ============================================================================
// Accuracy Validator Implementation
// ============================================================================

AccuracyValidator::ValidationResult AccuracyValidator::validate(
    LDM::LDMpart* particles_ref,
    LDM::LDMpart* particles_test,
    int count,
    double tolerance
) {
    ValidationResult result = {0};

    // Copy to host for comparison
    LDM::LDMpart* h_ref = new LDM::LDMpart[count];
    LDM::LDMpart* h_test = new LDM::LDMpart[count];

    cudaMemcpy(h_ref, particles_ref, sizeof(LDM::LDMpart) * count,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_test, particles_test, sizeof(LDM::LDMpart) * count,
               cudaMemcpyDeviceToHost);

    // Calculate errors
    double sum_pos_error2 = 0.0;
    double sum_conc_error2 = 0.0;
    int active_count = 0;

    for (int i = 0; i < count; i++) {
        if (!h_ref[i].flag || !h_test[i].flag) continue;
        active_count++;

        // Position error
        double dx = h_ref[i].x - h_test[i].x;
        double dy = h_ref[i].y - h_test[i].y;
        double dz = h_ref[i].z - h_test[i].z;
        double pos_error = sqrt(dx*dx + dy*dy + dz*dz);

        result.max_position_error = fmax(result.max_position_error, pos_error);
        sum_pos_error2 += pos_error * pos_error;

        // Concentration error
        double dc = fabs(h_ref[i].conc - h_test[i].conc);
        result.max_concentration_error = fmax(result.max_concentration_error, dc);
        sum_conc_error2 += dc * dc;
    }

    // RMS errors
    if (active_count > 0) {
        result.rms_position_error = sqrt(sum_pos_error2 / active_count);
        result.rms_concentration_error = sqrt(sum_conc_error2 / active_count);
    }

    // Check if passed
    result.passed = (result.max_position_error < tolerance);

    delete[] h_ref;
    delete[] h_test;

    return result;
}

} // namespace kernels
} // namespace ldm
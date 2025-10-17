// ldm_nuclides.cu - Nuclide Configuration Implementation
// Author: Juryong Park, 2025
#include "ldm_nuclides.cuh"
#include "../colors.h"

/**
 * @file ldm_nuclides.cu
 * @brief Implementation of nuclide configuration management
 * @author Juryong Park
 * @date 2025
 *
 * @details This file implements the NuclideConfig singleton class for
 *          managing radioactive nuclide properties and GPU memory.
 *
 * ## Key Functions
 *
 * - **loadFromFile()**: Parse CSV config and calculate derived properties
 * - **copyToDevice()**: Upload decay constants to GPU memory
 * - **Accessor methods**: Thread-safe property queries
 *
 * ## File Format Handling
 *
 * The CSV parser supports:
 * - Comment lines (starting with #)
 * - Empty lines (skipped)
 * - Scientific notation (e.g., 7.309e-10)
 * - Automatic normalization of initial ratios
 *
 * ## Decay Constant vs Half-Life
 *
 * The config file stores decay constants (λ), not half-lives:
 * - **Input**: λ [s⁻¹] from file
 * - **Calculated**: t½ = ln(2) / λ [hours]
 *
 * This choice optimizes for:
 * - Direct use in exponential decay: exp(-λt)
 * - Avoids repeated division in kernels
 * - More numerically stable for very long/short half-lives
 *
 * @see ldm_nuclides.cuh for class declaration
 * @see input/nuclides.conf for configuration file format
 */

// ============================================================================
// Static Member Initialization
// ============================================================================

/// Global singleton instance pointer (initialized to nullptr)
NuclideConfig* NuclideConfig::instance = nullptr;

// ============================================================================
// Constructor / Destructor
// ============================================================================

NuclideConfig::NuclideConfig()
    : num_nuclides(0), d_decay_constants(nullptr) {
}

NuclideConfig::~NuclideConfig() {
    if (d_decay_constants) {
        cudaFree(d_decay_constants);
    }
}

// ============================================================================
// Singleton Access
// ============================================================================

NuclideConfig* NuclideConfig::getInstance() {
    if (!instance) {
        instance = new NuclideConfig();
    }
    return instance;
}

// ============================================================================
// Load From File
// ============================================================================

bool NuclideConfig::loadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Cannot open nuclides config file: " << filename << std::endl;
        return false;
    }

    nuclides.clear();
    std::string line;

    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string name;
        float decay_const_from_file, ratio;

        // Parse CSV format: name,decay_constant,ratio
        std::string decay_str, ratio_str;

        if (std::getline(iss, name, ',') &&
            std::getline(iss, decay_str, ',') &&
            std::getline(iss, ratio_str)) {

            // Convert strings to floats
            try {
                decay_const_from_file = std::stof(decay_str);
                ratio = std::stof(ratio_str);
            } catch (const std::exception& e) {
                std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Failed to parse numeric values from line: " << line << std::endl;
                continue;
            }

            NuclideInfo info;
            strncpy(info.name, name.c_str(), 31);
            info.name[31] = '\0';

            // File contains decay constants (λ in s⁻¹), not half-lives
            // Take absolute value of the decay constant from file
            info.decay_constant = fabs(decay_const_from_file);

            // Calculate half-life from decay constant: t½ = ln(2) / λ
            // Result will be in seconds, convert to hours
            if (info.decay_constant > 0) {
                info.half_life_hours = 0.693147f / info.decay_constant / 3600.0f;
            } else {
                info.half_life_hours = 1e10f; // Very large value for stable nuclides
            }

            info.initial_ratio = ratio;

            nuclides.push_back(info);
        }
    }

    file.close();
    num_nuclides = nuclides.size();

    if (num_nuclides == 0) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "No nuclides loaded from config file" << std::endl;
        return false;
    }

    if (num_nuclides > MAX_NUCLIDES) {
        std::cerr << "[WARNING] Too many nuclides (" << num_nuclides
                 << "), limiting to " << MAX_NUCLIDES << std::endl;
        num_nuclides = MAX_NUCLIDES;
        nuclides.resize(MAX_NUCLIDES);
    }

    // Normalize initial ratios if they don't sum to 1
    float total_ratio = 0.0f;
    for (const auto& nuc : nuclides) {
        total_ratio += nuc.initial_ratio;
    }

    if (std::abs(total_ratio - 1.0f) > 0.001f) {
        for (auto& nuc : nuclides) {
            nuc.initial_ratio /= total_ratio;
        }
    }

    // Allocate and copy decay constants to device
    return copyToDevice();
}

// ============================================================================
// Copy To Device
// ============================================================================

bool NuclideConfig::copyToDevice() {
    if (d_decay_constants) {
        cudaFree(d_decay_constants);
    }

    // Prepare decay constants array
    std::vector<float> decay_constants(num_nuclides);
    for (int i = 0; i < num_nuclides; i++) {
        decay_constants[i] = nuclides[i].decay_constant;
    }

    // Allocate device memory
    cudaError_t err = cudaMalloc(&d_decay_constants, num_nuclides * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Failed to allocate device memory for decay constants: "
                 << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Copy to device
    err = cudaMemcpy(d_decay_constants, decay_constants.data(),
                    num_nuclides * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Failed to copy decay constants to device: "
                 << cudaGetErrorString(err) << std::endl;
        cudaFree(d_decay_constants);
        d_decay_constants = nullptr;
        return false;
    }

    // Try to copy to constant memory (may fail due to symbol visibility)
    err = cudaMemcpyToSymbol(d_decay_constants, decay_constants.data(),
                            num_nuclides * sizeof(float));
    if (err != cudaSuccess) {
        // Global memory fallback is already allocated and copied above
    }

    return true;
}

// ============================================================================
// Accessor Methods
// ============================================================================

float NuclideConfig::getDecayConstant(int index) const {
    if (index >= 0 && index < num_nuclides) {
        return nuclides[index].decay_constant;
    }
    return 0.0f;
}

float NuclideConfig::getInitialRatio(int index) const {
    if (index >= 0 && index < num_nuclides) {
        return nuclides[index].initial_ratio;
    }
    return 0.0f;
}

const char* NuclideConfig::getNuclideName(int index) const {
    if (index >= 0 && index < num_nuclides) {
        return nuclides[index].name;
    }
    return "Unknown";
}

float NuclideConfig::getHalfLife(int index) const {
    if (index >= 0 && index < num_nuclides) {
        return nuclides[index].half_life_hours;
    }
    return 0.0f;
}

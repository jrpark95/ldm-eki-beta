#pragma once
// ldm_nuclides.cuh - Nuclide Configuration and Decay Management
#ifndef LDM_NUCLIDES_CUH
#define LDM_NUCLIDES_CUH

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

/// Maximum number of nuclides supported
#define MAX_NUCLIDES 1

/// Decay constants array in constant memory
__constant__ float d_decay_constants[MAX_NUCLIDES];

/**
 * @struct NuclideInfo
 * @brief Information structure for a single nuclide
 *
 * @details Stores physical properties and initial conditions for
 *          radioactive nuclides in the simulation.
 */
struct NuclideInfo {
    char name[32];            ///< Nuclide name (e.g., "Cs-137")
    float half_life_hours;    ///< Half-life in hours
    float decay_constant;     ///< Decay constant λ [s⁻¹]
    float initial_ratio;      ///< Initial concentration ratio (normalized to 1.0)
};

/**
 * @class NuclideConfig
 * @brief Global nuclide configuration manager (singleton)
 *
 * @details Manages nuclide properties, decay constants, and GPU memory
 *          for radioactive decay calculations. Loads configuration from
 *          CSV files and uploads to device memory.
 *
 * @note Singleton pattern - use getInstance() to access
 * @note Decay constants automatically calculated from half-lives
 * @note Supports both constant memory and global memory fallback
 *
 * @usage
 * @code
 *   auto* config = NuclideConfig::getInstance();
 *   config->loadFromFile("input/nuclides_config_1.txt");
 *   float lambda = config->getDecayConstant(0);
 * @endcode
 */
class NuclideConfig {
private:
    static NuclideConfig* instance;     ///< Singleton instance
    std::vector<NuclideInfo> nuclides;  ///< Nuclide database
    int num_nuclides;                   ///< Number of loaded nuclides
    float* d_decay_constants;           ///< Device memory for decay constants

    /// Private constructor (singleton pattern)
    NuclideConfig();

public:
    /// Destructor - frees device memory
    ~NuclideConfig();

    /**
     * @brief Get singleton instance
     *
     * @return Pointer to global NuclideConfig instance
     *
     * @note Thread-safe for single-threaded initialization
     * @note Creates instance on first call
     */
    static NuclideConfig* getInstance();

    /**
     * @brief Load nuclide configuration from CSV file
     *
     * @details Parses CSV file with format: name,decay_constant,initial_ratio
     *          Calculates half-lives from decay constants.
     *          Normalizes initial ratios to sum to 1.0.
     *
     * @param[in] filename  Path to nuclide config file
     *
     * @return true if successful, false on file error or parse error
     *
     * @post Nuclides loaded into memory
     * @post Decay constants uploaded to device
     * @post Initial ratios normalized
     *
     * @note Lines starting with '#' are treated as comments
     * @note Empty lines are skipped
     * @note Decay constants from file are absolute values (|λ|)
     *
     * @warning If num_nuclides > MAX_NUCLIDES, excess nuclides are ignored
     *
     * @file_format
     * @code
     *   # Nuclide name, decay constant [s⁻¹], initial ratio
     *   Cs-137, 7.309e-10, 1.0
     * @endcode
     */
    bool loadFromFile(const std::string& filename);

    /**
     * @brief Copy decay constants to device memory
     *
     * @details Allocates device memory and copies decay constants.
     *          Attempts constant memory first, falls back to global memory.
     *
     * @return true if successful, false on CUDA error
     *
     * @post d_decay_constants allocated and populated
     * @post Constant memory symbol d_decay_constants updated (if available)
     *
     * @note Automatically called by loadFromFile()
     * @note Old device memory freed before reallocation
     */
    bool copyToDevice();

    /**
     * @brief Get number of loaded nuclides
     * @return Number of nuclides in configuration
     */
    int getNumNuclides() const { return num_nuclides; }

    /**
     * @brief Get all nuclide information
     * @return Const reference to nuclide vector
     */
    const std::vector<NuclideInfo>& getNuclides() const { return nuclides; }

    /**
     * @brief Get device memory pointer for decay constants
     * @return Device pointer to decay constants array
     * @note For passing to CUDA kernels
     */
    float* getDeviceDecayConstants() { return d_decay_constants; }

    /**
     * @brief Get decay constant for specific nuclide
     *
     * @param[in] index  Nuclide index (0 to num_nuclides-1)
     *
     * @return Decay constant λ [s⁻¹], or 0.0 if index out of range
     *
     * @equation λ = ln(2) / t_{1/2}
     */
    float getDecayConstant(int index) const;

    /**
     * @brief Get initial ratio for specific nuclide
     *
     * @param[in] index  Nuclide index (0 to num_nuclides-1)
     *
     * @return Initial concentration ratio (normalized), or 0.0 if out of range
     *
     * @note Sum of all ratios equals 1.0 after normalization
     */
    float getInitialRatio(int index) const;

    /**
     * @brief Get nuclide name
     *
     * @param[in] index  Nuclide index (0 to num_nuclides-1)
     *
     * @return Nuclide name string, or "Unknown" if out of range
     */
    const char* getNuclideName(int index) const;

    /**
     * @brief Get half-life for specific nuclide
     *
     * @param[in] index  Nuclide index (0 to num_nuclides-1)
     *
     * @return Half-life in hours, or 0.0 if out of range
     *
     * @note Calculated from decay constant: t_{1/2} = ln(2) / λ / 3600
     */
    float getHalfLife(int index) const;
};

#endif // LDM_NUCLIDES_CUH

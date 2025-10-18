#pragma once
// ldm_nuclides.cuh - Nuclide Configuration and Decay Management
// Author: Juryong Park, 2025
#ifndef LDM_NUCLIDES_CUH
#define LDM_NUCLIDES_CUH

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

/**
 * @file ldm_nuclides.cuh
 * @brief Nuclide configuration management for radioactive decay simulations
 * @author Juryong Park
 * @date 2025
 *
 * @details This module manages nuclide properties, decay constants, and
 *          initial conditions for radioactive decay simulations. It provides
 *          a singleton interface for loading nuclide configurations from
 *          CSV files and uploading decay constants to GPU memory.
 *
 * ## Nuclide Properties
 *
 * Each nuclide is characterized by:
 * - Name: Isotope identifier (e.g., "Cs-137", "I-131")
 * - Half-life: Time for 50% decay (hours)
 * - Decay constant: λ = ln(2) / t½ (s⁻¹)
 * - Initial ratio: Normalized initial concentration (Σ ratios = 1.0)
 *
 * ## Decay Constant Calculation
 *
 * The decay constant λ relates to half-life t½ by:
 * @equation
 *   λ = ln(2) / t½
 *
 * Activity decreases exponentially:
 * @equation
 *   A(t) = A₀ × exp(-λt)
 *
 * ## File Format
 *
 * CSV format: name,decay_constant,initial_ratio
 * @code
 *   # Comment lines start with #
 *   Cs-137, 7.309e-10, 1.0
 *   I-131, 9.978e-07, 0.5
 * @endcode
 *
 * @note For complex decay chains (multiple nuclides), use CRAM system instead
 * @see ldm_cram2.cuh for multi-nuclide decay chains
 * @see input/nuclides.conf for configuration file format
 */

/// Maximum number of nuclides supported by constant memory
/// @note Limited by __constant__ memory size (64 KB total)
/// @note For larger chains, use global memory or CRAM system
#define MAX_NUCLIDES 1

/// Decay constants array in device constant memory [s⁻¹]
/// @note Faster access than global memory for small arrays
/// @note May not be visible in non-RDC compilation (fallback to global memory)
__constant__ float d_decay_constants[MAX_NUCLIDES];

/**
 * @struct NuclideInfo
 * @brief Physical and chemical properties of a radioactive nuclide
 *
 * @details Stores all information needed to simulate radioactive decay
 *          and transport of a single nuclide species. This structure
 *          is used for simple single-nuclide simulations.
 *
 * ## Physical Interpretation
 *
 * - Half-life: Time for activity to drop to 50% of initial value
 * - Decay constant: Exponential decay rate (λ = ln(2)/t½)
 * - Initial ratio: Fraction of total activity from this nuclide
 *
 * ## Example Values (Common Isotopes)
 *
 * | Nuclide | Half-life      | Decay constant λ (s⁻¹) |
 * |---------|----------------|-------------------------|
 * | I-131   | 8.02 days      | 9.978e-7                |
 * | Cs-137  | 30.17 years    | 7.309e-10               |
 * | Xe-133  | 5.24 days      | 1.530e-6                |
 * | Te-132  | 3.20 days      | 2.508e-6                |
 *
 * @note For multi-nuclide chains with production/decay, use CRAM system
 * @see ldm_cram2.cuh for coupled decay chain simulations
 */
struct NuclideInfo {
    char name[32];            ///< Isotope name (e.g., "Cs-137", "I-131")
    float half_life_hours;    ///< Half-life in hours (calculated from λ)
    float decay_constant;     ///< Decay constant λ [s⁻¹], primary physical parameter
    float initial_ratio;      ///< Initial concentration fraction (normalized: Σ = 1.0)
};

/**
 * @class NuclideConfig
 * @brief Global nuclide configuration manager (singleton pattern)
 *
 * @details Centralized management of nuclide properties and GPU memory
 *          for radioactive decay calculations. This class provides:
 *
 * ## Features
 *
 * - Singleton access: Single global instance via getInstance()
 * - CSV loading: Parse nuclide config files (name, λ, ratio)
 * - Automatic normalization: Initial ratios summed to 1.0
 * - GPU memory management: Upload decay constants to device
 * - Constant memory fallback: Tries __constant__, falls back to global
 *
 * ## Design Rationale
 *
 * Singleton pattern ensures:
 * - Consistent nuclide properties across all simulation modules
 * - Single source of truth for decay constants
 * - Automatic GPU memory lifecycle management
 * - Global access without passing pointers
 *
 * ## Typical Usage
 *
 * @code
 *   // Initialize (called once at startup)
 *   auto* config = NuclideConfig::getInstance();
 *   if (!config->loadFromFile("input/nuclides.conf")) {
 *       std::cerr << "Failed to load nuclides\n";
 *       return -1;
 *   }
 *
 *   // Query properties (anywhere in code)
 *   std::cout << "Nuclide: " << config->getNuclideName(0) << "\n";
 *   std::cout << "Half-life: " << config->getHalfLife(0) << " hours\n";
 *   std::cout << "Decay constant: " << config->getDecayConstant(0) << " s⁻¹\n";
 *
 *   // Device pointer for kernels
 *   float* d_lambda = config->getDeviceDecayConstants();
 * @endcode
 *
 * ## Memory Lifecycle
 *
 * - Creation: First call to getInstance() allocates singleton
 * - Loading: loadFromFile() allocates device memory
 * - Destruction: ~NuclideConfig() frees device memory
 * - Cleanup: Automatic at program exit
 *
 * @note Thread-safe for single-threaded initialization (no mutex)
 * @note Device memory freed automatically in destructor
 * @note Initial ratios normalized even if file values don't sum to 1.0
 *
 * @warning Do not create NuclideConfig directly - use getInstance()
 * @warning Loading new file frees old device memory
 *
 * @see ldm_cram2.cuh for complex multi-nuclide decay chains
 * @see input/nuclides.conf for file format
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

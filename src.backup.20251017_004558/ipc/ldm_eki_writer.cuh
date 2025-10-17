#pragma once
// ldm_eki_writer.cuh - IPC Writer for LDM→Python communication
#ifndef LDM_EKI_WRITER_CUH
#define LDM_EKI_WRITER_CUH

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include "colors.h"

// Forward declarations
class MemoryDoctor;
extern MemoryDoctor g_memory_doctor;

// Forward declaration of EKIConfig (defined in ldm_struct.cuh)
struct EKIConfig;

namespace LDM_EKI_IPC {

// ============================================================================
// Shared Memory Configuration Names
// ============================================================================
constexpr const char* SHM_CONFIG_NAME = "/ldm_eki_config";
constexpr const char* SHM_DATA_NAME = "/ldm_eki_data";
constexpr const char* SHM_ENSEMBLE_OBS_CONFIG_NAME = "/ldm_eki_ensemble_obs_config";
constexpr const char* SHM_ENSEMBLE_OBS_DATA_NAME = "/ldm_eki_ensemble_obs_data";

// ============================================================================
// Configuration Structures
// ============================================================================

/**
 * @brief Basic EKI configuration structure (12 bytes)
 * @details Minimal configuration for backward compatibility
 */
struct EKIConfigBasic {
    int32_t ensemble_size;  ///< Number of ensemble members
    int32_t num_receptors;  ///< Number of receptors
    int32_t num_timesteps;  ///< Number of time steps
};

/**
 * @brief Full EKI configuration structure (128 bytes exactly)
 * @details Complete configuration including algorithm parameters and options
 *
 * @note Total size is exactly 128 bytes for efficient memory alignment
 */
struct EKIConfigFull {
    // Basic info (12 bytes)
    int32_t ensemble_size;
    int32_t num_receptors;
    int32_t num_timesteps;

    // Algorithm parameters (44 bytes)
    int32_t iteration;
    float renkf_lambda;
    float noise_level;
    float time_days;
    float time_interval;          ///< EKI time interval (e.g., 15.0 minutes)
    float inverse_time_interval;
    float receptor_error;
    float receptor_mda;
    float prior_constant;         ///< Prior emission constant value (e.g., 1.5e+8)
    int32_t num_source;
    int32_t num_gpu;

    // Option strings (64 bytes = 8 strings × 8 bytes)
    char perturb_option[8];    ///< "Off"
    char adaptive_eki[8];      ///< "Off"
    char localized_eki[8];     ///< "Off"
    char regularization[8];    ///< "On"
    char gpu_forward[8];       ///< "On"
    char gpu_inverse[8];       ///< "On"
    char source_location[8];   ///< "Fixed"
    char time_unit[8];         ///< "minutes"

    // Memory Doctor Mode (8 bytes)
    char memory_doctor[8];     ///< "On" or "Off"

    // Total: 12 + 44 + 64 + 8 = 128 bytes (no padding needed)
};

/**
 * @brief Data header structure for observation data (12 bytes + data)
 */
struct EKIDataHeader {
    int32_t status;      ///< 0=writing, 1=ready
    int32_t rows;        ///< Number of receptors
    int32_t cols;        ///< Number of timesteps
    // float data[] follows immediately after header
};

// ============================================================================
// EKIWriter Class - Write observation data from C++ to Python
// ============================================================================

/**
 * @class EKIWriter
 * @brief Manages shared memory IPC for writing observation data to Python
 *
 * @details Handles POSIX shared memory creation and data transfer from the
 *          LDM C++/CUDA simulation to the Python EKI inversion process.
 *          Supports both initial observations and ensemble observations.
 *
 * @note Thread-safe for single writer, multiple readers
 * @note All shared memory segments use /dev/shm/ (tmpfs)
 *
 * @see EKIReader for the reciprocal Python→C++ communication
 */
class EKIWriter {
private:
    int config_fd;         ///< File descriptor for configuration segment
    int data_fd;           ///< File descriptor for data segment
    void* config_map;      ///< Memory-mapped configuration structure
    void* data_map;        ///< Memory-mapped data buffer
    size_t data_size;      ///< Size of data segment in bytes
    bool initialized;      ///< Initialization state

public:
    EKIWriter();
    ~EKIWriter();

    /**
     * @brief Initialize shared memory segments with full configuration
     *
     * @details Creates and maps POSIX shared memory for both configuration
     *          and observation data. Configuration includes all EKI algorithm
     *          parameters needed by Python process.
     *
     * @param[in] eki_config     Complete EKI configuration structure
     * @param[in] num_timesteps  Number of simulation timesteps
     *
     * @return true if initialization successful, false otherwise
     *
     * @pre POSIX shared memory support available (/dev/shm mounted)
     * @post Shared memory segments created and mapped
     * @post Configuration data written to shared memory
     *
     * @note Automatically calculates data segment size from config
     * @warning Must be called before any write operations
     */
    bool initialize(const ::EKIConfig& eki_config, int num_timesteps);

    /**
     * @brief Write initial observation data to shared memory
     *
     * @details Transfers receptor observation matrix (receptors × timesteps)
     *          from C++ to Python via shared memory. Used for initial "truth"
     *          simulation observations.
     *
     * @param[in] observations  Observation matrix (row-major, receptors × timesteps)
     * @param[in] rows          Number of receptors
     * @param[in] cols          Number of timesteps
     *
     * @return true if write successful, false otherwise
     *
     * @pre initialize() must have been called successfully
     * @pre Dimensions must match those provided to initialize()
     *
     * @note Sets status flag to signal data ready to Python
     * @note Logs data to Memory Doctor if enabled
     *
     * @see initializeEnsembleObservations() for ensemble data
     */
    bool writeObservations(const float* observations, int rows, int cols);

    /**
     * @brief Initialize ensemble observation shared memory segments
     *
     * @details Creates separate shared memory segments for ensemble observation
     *          configuration and data. Must be called before writeEnsembleObservations().
     *
     * @param[in] ensemble_size   Number of ensemble members
     * @param[in] num_receptors   Number of receptors
     * @param[in] num_timesteps   Number of timesteps
     *
     * @return true if successful, false otherwise
     *
     * @pre initialize() must have been called
     * @post Ensemble observation config segment created
     *
     * @note Creates /ldm_eki_ensemble_obs_config segment
     */
    bool initializeEnsembleObservations(int ensemble_size, int num_receptors, int num_timesteps);

    /**
     * @brief Write ensemble observations to shared memory
     *
     * @details Transfers ensemble observation tensor (ensemble × receptors × timesteps)
     *          from C++ to Python. Used for each EKI iteration.
     *
     * @param[in] observations    3D observation tensor (row-major order)
     * @param[in] ensemble_size   Number of ensemble members
     * @param[in] num_receptors   Number of receptors
     * @param[in] num_timesteps   Number of timesteps
     * @param[in] iteration       Current EKI iteration number (-1 if not applicable)
     *
     * @return true if write successful, false otherwise
     *
     * @pre initializeEnsembleObservations() must have been called
     *
     * @note Calculates and logs statistics (min/max/mean) for validation
     * @note Logs to Memory Doctor with iteration tracking if enabled
     *
     * @performance Transfers ~10-100 KB per iteration typically
     */
    bool writeEnsembleObservations(const float* observations, int ensemble_size,
                                   int num_receptors, int num_timesteps, int iteration = -1);

    /**
     * @brief Get current basic configuration
     *
     * @param[out] ensemble_size   Number of ensemble members
     * @param[out] num_receptors   Number of receptors
     * @param[out] num_timesteps   Number of timesteps
     *
     * @return true if config available, false if not initialized
     */
    bool getConfig(int& ensemble_size, int& num_receptors, int& num_timesteps);

    /**
     * @brief Cleanup resources and unmap memory
     *
     * @details Unmaps all memory-mapped segments and closes file descriptors.
     *          Does not unlink shared memory files (see unlinkSharedMemory()).
     *
     * @post All file descriptors closed
     * @post All memory mappings released
     * @post initialized flag set to false
     *
     * @note Safe to call multiple times
     * @note Automatically called by destructor
     */
    void cleanup();

    /**
     * @brief Unlink shared memory segments
     *
     * @details Removes shared memory files from /dev/shm. Should be called
     *          at program exit after all readers have finished.
     *
     * @note Static method - can be called without instance
     * @note Does NOT unlink ensemble observation files (Python needs them)
     * @warning Call only when certain no processes need the data
     */
    static void unlinkSharedMemory();
};

} // namespace LDM_EKI_IPC

#endif // LDM_EKI_WRITER_CUH

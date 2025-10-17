#pragma once
// ldm_eki_reader.cuh - IPC Reader for Python→LDM communication
#ifndef LDM_EKI_READER_CUH
#define LDM_EKI_READER_CUH

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include "colors.h"

// Forward declarations
class MemoryDoctor;
extern MemoryDoctor g_memory_doctor;

namespace LDM_EKI_IPC {

// ============================================================================
// Shared Memory Configuration Names for Ensemble Data (Python → C++)
// ============================================================================
constexpr const char* SHM_ENSEMBLE_CONFIG_NAME = "/ldm_eki_ensemble_config";
constexpr const char* SHM_ENSEMBLE_DATA_NAME = "/ldm_eki_ensemble_data";

// ============================================================================
// Ensemble Configuration Structures
// ============================================================================

/**
 * @brief Ensemble configuration structure (12 bytes)
 * @details Configuration for ensemble state data transfer from Python to C++
 */
struct EnsembleConfig {
    int32_t num_states;      ///< Number of state variables (e.g., 24 timesteps)
    int32_t num_ensemble;    ///< Number of ensemble members (e.g., 100)
    int32_t timestep_id;     ///< Current timestep identifier (iteration tracking)
};

/**
 * @brief Ensemble data header structure (12 bytes + data)
 */
struct EnsembleDataHeader {
    int32_t status;          ///< 0=writing, 1=ready
    int32_t rows;            ///< Number of states
    int32_t cols;            ///< Number of ensemble members
    // float data[] follows immediately after header
};

// ============================================================================
// EKIReader Class - Read ensemble state data from Python
// ============================================================================

/**
 * @class EKIReader
 * @brief Manages shared memory IPC for reading ensemble state data from Python
 *
 * @details Handles POSIX shared memory access for reading ensemble state
 *          updates from the Python EKI optimization process. Each iteration,
 *          Python writes updated ensemble states and C++ reads them to run
 *          the next forward simulation.
 *
 * @note Thread-safe for single writer (Python), single reader (C++)
 * @note Uses fresh data detection via iteration ID to prevent stale reads
 *
 * @see EKIWriter for the reciprocal C++→Python communication
 */
class EKIReader {
private:
    int config_fd;         ///< File descriptor for configuration segment
    int data_fd;           ///< File descriptor for data segment
    void* config_map;      ///< Memory-mapped configuration structure
    void* data_map;        ///< Memory-mapped data buffer
    size_t data_size;      ///< Size of data segment in bytes
    bool initialized;      ///< Initialization state

public:
    EKIReader();
    ~EKIReader();

    /**
     * @brief Wait for ensemble data to become available
     *
     * @details Polls shared memory files until fresh ensemble data appears
     *          from Python. Uses iteration ID to detect new vs stale data.
     *          Timeout prevents infinite waiting on Python failures.
     *
     * @param[in] timeout_seconds      Maximum time to wait [seconds]
     * @param[in] expected_iteration   Expected iteration number (-1 to ignore)
     *
     * @return true if fresh data detected, false on timeout
     *
     * @note Checks /dev/shm/ldm_eki_ensemble_config and _data
     * @note Prints progress messages every 5 seconds
     *
     * @warning Blocking call - will wait up to timeout_seconds
     *
     * @performance Polls every 1 second, negligible CPU usage
     */
    bool waitForEnsembleData(int timeout_seconds = 60, int expected_iteration = -1);

    /**
     * @brief Read ensemble configuration from shared memory
     *
     * @details Reads the 12-byte ensemble configuration header containing
     *          dimensions and iteration tracking information.
     *
     * @param[out] num_states    Number of state variables
     * @param[out] num_ensemble  Number of ensemble members
     * @param[out] timestep_id   Current iteration identifier
     *
     * @return true if read successful, false otherwise
     *
     * @note Opens, reads, and closes file in one operation
     * @note Does not keep config file descriptor open
     */
    bool readEnsembleConfig(int& num_states, int& num_ensemble, int& timestep_id);

    /**
     * @brief Read ensemble state matrix from shared memory
     *
     * @details Reads the complete ensemble state tensor (states × ensembles)
     *          from Python. Validates dimensions, status, and data integrity.
     *          Calculates statistics for debugging.
     *
     * @param[out] output        Output vector to receive state data
     * @param[out] num_states    Number of state variables (from config)
     * @param[out] num_ensemble  Number of ensemble members (from config)
     *
     * @return true if read successful, false otherwise
     *
     * @pre waitForEnsembleData() returned true
     * @post output vector resized and populated
     * @post Statistics logged to console
     *
     * @note Calls readEnsembleConfig() internally
     * @note Logs to Memory Doctor with iteration tracking if enabled
     * @note Memory mapping automatically unmapped after read
     *
     * @performance Typical read: 10-100 KB in < 1ms
     */
    bool readEnsembleStates(std::vector<float>& output, int& num_states, int& num_ensemble);

    /**
     * @brief Cleanup resources and unmap memory
     *
     * @details Unmaps all memory-mapped segments and closes file descriptors.
     *          Does not unlink shared memory files (see unlinkEnsembleSharedMemory()).
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
     * @brief Unlink ensemble shared memory segments
     *
     * @details Removes ensemble shared memory files from /dev/shm.
     *          Should be called at program exit after all operations complete.
     *
     * @note Static method - can be called without instance
     * @warning Call only when certain no processes need the data
     */
    static void unlinkEnsembleSharedMemory();
};

} // namespace LDM_EKI_IPC

#endif // LDM_EKI_READER_CUH

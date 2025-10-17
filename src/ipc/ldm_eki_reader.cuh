////////////////////////////////////////////////////////////////////////////////
/// @file    ldm_eki_reader.cuh
/// @brief   IPC reader for receiving ensemble state data from Python
/// @details Manages POSIX shared memory segments to receive updated ensemble
///          states from the Python EKI optimization process. Each iteration,
///          Python writes optimized emission states and C++ reads them to run
///          the next forward simulation.
///
/// @author  Juryong Park
/// @date    2025
////////////////////////////////////////////////////////////////////////////////

#pragma once
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

////////////////////////////////////////////////////////////////////////////////
/// @name Shared Memory Segment Names for Ensemble Data (Python → C++)
/// @{
/// @note All segments reside in /dev/shm (tmpfs)
////////////////////////////////////////////////////////////////////////////////

constexpr const char* SHM_ENSEMBLE_CONFIG_NAME = "/ldm_eki_ensemble_config";
constexpr const char* SHM_ENSEMBLE_DATA_NAME = "/ldm_eki_ensemble_data";

/// @}

////////////////////////////////////////////////////////////////////////////////
/// @name Ensemble Configuration Structures
/// @{
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// @struct EnsembleConfig
/// @brief  Ensemble configuration structure (12 bytes)
/// @details Configuration for ensemble state data transfer from Python to C++
///
/// Memory Layout:
/// - Bytes 0-3  : num_states (number of state variables, e.g., 24 timesteps)
/// - Bytes 4-7  : num_ensemble (number of ensemble members, e.g., 100)
/// - Bytes 8-11 : timestep_id (iteration identifier for freshness detection)
////////////////////////////////////////////////////////////////////////////////
struct EnsembleConfig {
    int32_t num_states;      ///< Number of state variables (e.g., 24 emission timesteps)
    int32_t num_ensemble;    ///< Number of ensemble members (e.g., 100)
    int32_t timestep_id;     ///< Current iteration ID (for stale data detection)
};

////////////////////////////////////////////////////////////////////////////////
/// @struct EnsembleDataHeader
/// @brief  Ensemble data header structure (12 bytes + variable data)
/// @details Prepends ensemble state data to provide metadata and ready status
///
/// Memory Layout:
/// - [0-11 bytes]      : Header (status, dimensions)
/// - [12+ bytes]       : Float data (rows × cols elements)
////////////////////////////////////////////////////////////////////////////////
struct EnsembleDataHeader {
    int32_t status;          ///< 0=writing (incomplete), 1=ready (complete)
    int32_t rows;            ///< Number of states
    int32_t cols;            ///< Number of ensemble members
    // float data[] follows immediately after header
};

/// @}

////////////////////////////////////////////////////////////////////////////////
/// @class EKIReader
/// @brief IPC reader for receiving ensemble state data from Python
///
/// @details
/// This class manages POSIX shared memory access for reading ensemble state
/// updates from the Python EKI optimization process. It handles the reciprocal
/// data flow to EKIWriter: while EKIWriter sends observations from C++ to Python,
/// EKIReader receives optimized emission states from Python to C++.
///
/// Communication Protocol:
/// ```
/// 1. C++ calls waitForEnsembleData() → polls for fresh data from Python
/// 2. Python writes ensemble states → sets status=1 and timestep_id
/// 3. C++ detects fresh data (new timestep_id) → proceeds to read
/// 4. C++ calls readEnsembleStates() → copies data to host memory
/// 5. C++ runs forward simulation with new emission states
/// ```
///
/// Freshness Detection:
/// - Uses timestep_id to distinguish new vs stale data
/// - Prevents re-reading same iteration multiple times
/// - Static variable tracks last processed iteration
///
/// Shared Memory Segments:
/// - `/dev/shm/ldm_eki_ensemble_config` : Configuration (12 bytes)
/// - `/dev/shm/ldm_eki_ensemble_data` : State data (header + data)
///
/// Data Format (Row-Major):
/// - Ensemble states: [states × ensembles] matrix
/// - Example: [24 timesteps × 100 ensembles] = 2400 floats = 9.6 KB
///
/// @note Thread-safe for single writer (Python), single reader (this class)
/// @note Polling-based with 1-second intervals (low CPU usage)
///
/// @see EKIWriter for reciprocal C++→Python data transfer
/// @see Memory Doctor system for IPC debugging and validation
///
/// @par Example Usage:
/// @code
/// EKIReader reader;
/// if (reader.waitForEnsembleData(60, expected_iteration)) {
///     std::vector<float> states;
///     int num_states, num_ensemble;
///     reader.readEnsembleStates(states, num_states, num_ensemble);
///     // Use states for forward simulation...
/// }
/// reader.cleanup();
/// @endcode
////////////////////////////////////////////////////////////////////////////////
class EKIReader {
private:
    int config_fd;         ///< File descriptor for configuration segment
    int data_fd;           ///< File descriptor for data segment
    void* config_map;      ///< Memory-mapped configuration structure (EnsembleConfig*)
    void* data_map;        ///< Memory-mapped data buffer (EnsembleDataHeader + float[])
    size_t data_size;      ///< Size of data segment [bytes]
    bool initialized;      ///< Initialization state flag

public:
    ////////////////////////////////////////////////////////////////////////////////
    /// @name Constructor / Destructor
    /// @{
    ////////////////////////////////////////////////////////////////////////////////

    EKIReader();
    ~EKIReader();

    /// @}

    ////////////////////////////////////////////////////////////////////////////////
    /// @name Waiting for Data
    /// @{
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    /// @brief Wait for fresh ensemble data to become available
    ///
    /// @details
    /// Polls shared memory files until fresh ensemble data appears from Python.
    /// Uses timestep_id (iteration identifier) to distinguish new vs stale data.
    /// This prevents re-reading the same iteration multiple times.
    ///
    /// Freshness Detection:
    /// - Static variable tracks last_iteration_id
    /// - Compares config.timestep_id > last_iteration_id
    /// - Only returns true when new iteration detected
    ///
    /// Polling Strategy:
    /// - Checks every 1 second (sleep(1))
    /// - Prints status every 5 seconds
    /// - Returns false on timeout
    ///
    /// @param[in] timeout_seconds      Maximum time to wait [seconds] (default: 60)
    /// @param[in] expected_iteration   Expected iteration number (unused, for future)
    ///
    /// @return true if fresh data detected, false on timeout
    ///
    /// @note Checks /dev/shm/ldm_eki_ensemble_config and _data
    /// @note Blocking call - will wait up to timeout_seconds
    ///
    /// @warning Timeout may indicate Python process failure
    ///
    /// @par Output:
    /// Prints waiting status and fresh data detection message
    ///
    /// @performance CPU usage negligible (~0.01% during polling)
    ////////////////////////////////////////////////////////////////////////////////
    bool waitForEnsembleData(int timeout_seconds = 60, int expected_iteration = -1);

    /// @}

    ////////////////////////////////////////////////////////////////////////////////
    /// @name Reading Configuration and Data
    /// @{
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    /// @brief Read ensemble configuration from shared memory
    ///
    /// @details
    /// Reads the 12-byte ensemble configuration header containing dimensions
    /// and iteration tracking information. This is called internally by
    /// readEnsembleStates() but can also be used standalone.
    ///
    /// Steps:
    /// 1. Open /dev/shm/ldm_eki_ensemble_config
    /// 2. Read EnsembleConfig structure
    /// 3. Close file descriptor immediately
    /// 4. Extract dimensions and iteration ID
    ///
    /// @param[out] num_states    Number of state variables (e.g., 24 timesteps)
    /// @param[out] num_ensemble  Number of ensemble members (e.g., 100)
    /// @param[out] timestep_id   Current iteration identifier
    ///
    /// @return true if read successful, false on error
    ///
    /// @note Opens, reads, and closes file in one operation (stateless)
    /// @note Does not keep config file descriptor open
    ///
    /// @par Output:
    /// Prints configuration dimensions and iteration ID
    ////////////////////////////////////////////////////////////////////////////////
    bool readEnsembleConfig(int& num_states, int& num_ensemble, int& timestep_id);

    ////////////////////////////////////////////////////////////////////////////////
    /// @brief Read ensemble state matrix from shared memory
    ///
    /// @details
    /// Reads the complete ensemble state tensor (states × ensembles) from Python.
    /// Validates dimensions, status, and data integrity. Calculates statistics
    /// for debugging and validation.
    ///
    /// Data Layout (Row-Major):
    /// ```
    /// [state_0_ens_0, state_0_ens_1, ..., state_0_ens_N,
    ///  state_1_ens_0, state_1_ens_1, ..., state_1_ens_N,
    ///  ...,
    ///  state_S_ens_0, state_S_ens_1, ..., state_S_ens_N]
    /// ```
    ///
    /// Steps:
    /// 1. Call readEnsembleConfig() to get dimensions
    /// 2. Open and stat data file to verify size
    /// 3. Memory-map data segment
    /// 4. Read and validate header
    /// 5. Copy data to output vector with memcpy
    /// 6. Calculate statistics (min/max/mean)
    /// 7. Log to Memory Doctor if enabled
    /// 8. Unmap and close file
    ///
    /// @param[out] output        Output vector to receive state data
    /// @param[out] num_states    Number of state variables (from config)
    /// @param[out] num_ensemble  Number of ensemble members (from config)
    ///
    /// @return true if read successful, false on any error
    ///
    /// @pre waitForEnsembleData() returned true
    /// @post output vector resized to [num_states × num_ensemble]
    /// @post Statistics logged to console
    ///
    /// @note Calls readEnsembleConfig() internally
    /// @note Memory mapping automatically unmapped after read
    /// @note Logs to Memory Doctor with iteration tracking if enabled
    ///
    /// @par Output:
    /// Prints data size, dimensions, and statistics (range, mean)
    ///
    /// @performance Typical read: 10-100 KB in < 1ms
    ////////////////////////////////////////////////////////////////////////////////
    bool readEnsembleStates(std::vector<float>& output, int& num_states, int& num_ensemble);

    /// @}

    ////////////////////////////////////////////////////////////////////////////////
    /// @name Cleanup
    /// @{
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    /// @brief Cleanup resources and unmap memory
    ///
    /// @details
    /// Unmaps all memory-mapped segments and closes file descriptors.
    /// Does NOT unlink shared memory files from /dev/shm - use
    /// unlinkEnsembleSharedMemory() for that.
    ///
    /// Steps:
    /// 1. Unmap data segment if mapped
    /// 2. Unmap config segment if mapped
    /// 3. Close data file descriptor
    /// 4. Close config file descriptor
    /// 5. Set initialized flag to false
    ///
    /// @post All file descriptors closed
    /// @post All memory mappings released
    /// @post initialized flag set to false
    /// @post Shared memory files remain in /dev/shm
    ///
    /// @note Safe to call multiple times (idempotent)
    /// @note Automatically called by destructor
    /// @note Does NOT delete shared memory files from filesystem
    ////////////////////////////////////////////////////////////////////////////////
    void cleanup();

    ////////////////////////////////////////////////////////////////////////////////
    /// @brief Unlink ensemble shared memory segments from filesystem
    ///
    /// @details
    /// Removes ensemble shared memory files from /dev/shm. Should be called
    /// at program exit after all operations complete.
    ///
    /// Unlinks:
    /// - /dev/shm/ldm_eki_ensemble_config
    /// - /dev/shm/ldm_eki_ensemble_data
    ///
    /// @note Static method - can be called without instance
    /// @warning Call only when certain no processes need the data
    /// @warning Premature unlinking will cause read failures
    ///
    /// @par Output:
    /// Prints confirmation message to console
    ////////////////////////////////////////////////////////////////////////////////
    static void unlinkEnsembleSharedMemory();

    /// @}
};

} // namespace LDM_EKI_IPC

#endif // LDM_EKI_READER_CUH

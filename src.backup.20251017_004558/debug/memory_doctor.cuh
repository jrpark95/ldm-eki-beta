#pragma once
// memory_doctor.cuh - Memory Doctor Mode for IPC debugging
#ifndef MEMORY_DOCTOR_CUH
#define MEMORY_DOCTOR_CUH

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdint>
#include "colors.h"

/**
 * @class MemoryDoctor
 * @brief Diagnostic tool for debugging inter-process communication data transfer
 *
 * @details Memory Doctor mode logs all IPC data transfers between C++ and Python
 *          processes, including checksums, statistics, and sample data. Useful for
 *          diagnosing communication issues, data corruption, or dimension mismatches.
 *
 * @note All log files written to logs/memory_doctor/ directory
 * @note Automatically cleans previous logs when enabled
 * @note Logs include iteration tracking for temporal analysis
 *
 * @warning Generates significant I/O overhead - use only for debugging
 * @warning Log files can grow large for many iterations
 *
 * @usage
 * @code
 *   g_memory_doctor.setEnabled(true);  // Enable mode
 *   // ... perform IPC operations ...
 *   // Check logs/memory_doctor/iter###_*.txt files
 * @endcode
 */
class MemoryDoctor {
private:
    bool enabled;                ///< Whether Memory Doctor mode is active
    std::string log_dir;         ///< Directory for log files

    /**
     * @brief Calculate simple checksum for data verification
     *
     * @param[in] data   Float array to checksum
     * @param[in] count  Number of elements
     *
     * @return 32-bit checksum value
     *
     * @note Uses XOR and rotate-left for simple error detection
     * @note Not cryptographically secure - for debugging only
     */
    uint32_t calculateChecksum(const float* data, size_t count) const;

    /**
     * @brief Clean all previous log files in memory_doctor directory
     *
     * @details Removes all .txt files from logs/memory_doctor/ to prevent
     *          confusion between runs.
     *
     * @note Called automatically when setEnabled(true)
     */
    void cleanLogDirectory();

public:
    MemoryDoctor();

    /**
     * @brief Enable or disable Memory Doctor mode
     *
     * @param[in] enable  true to enable, false to disable
     *
     * @post If enabling: log directory created and cleaned
     * @post Mode status printed to console
     *
     * @note Creates logs/ and logs/memory_doctor/ if they don't exist
     */
    void setEnabled(bool enable);

    /**
     * @brief Check if Memory Doctor mode is enabled
     *
     * @return true if enabled, false otherwise
     */
    bool isEnabled() const { return enabled; }

    /**
     * @brief Log data being sent from LDM to Python
     *
     * @details Records complete diagnostic information about data transfer,
     *          including dimensions, statistics, checksums, and sample data.
     *
     * @param[in] data_type    Descriptive name (e.g., "initial_observations")
     * @param[in] data         Float array being sent
     * @param[in] rows         Number of rows in matrix
     * @param[in] cols         Number of columns in matrix
     * @param[in] iteration    Current EKI iteration number
     * @param[in] extra_info   Additional context information
     *
     * @note Creates file: logs/memory_doctor/iter###_cpp_sent_{data_type}.txt
     * @note Logs first 100 and last 100 elements
     * @note Calculates min/max/mean, zero count, NaN/Inf counts
     *
     * @performance ~1-2ms overhead per call typical
     */
    void logSentData(const std::string& data_type, const float* data,
                     int rows, int cols, int iteration = -1,
                     const std::string& extra_info = "");

    /**
     * @brief Log data received by LDM from Python
     *
     * @details Records complete diagnostic information about received data,
     *          enabling comparison with Python's sent data logs.
     *
     * @param[in] data_type    Descriptive name (e.g., "ensemble_states")
     * @param[in] data         Float array received
     * @param[in] rows         Number of rows in matrix
     * @param[in] cols         Number of columns in matrix
     * @param[in] iteration    Current EKI iteration number
     * @param[in] extra_info   Additional context information
     *
     * @note Creates file: logs/memory_doctor/iter###_cpp_recv_{data_type}.txt
     * @note Format matches logSentData() for easy comparison
     *
     * @see logSentData() for detailed logging format
     */
    void logReceivedData(const std::string& data_type, const float* data,
                        int rows, int cols, int iteration = -1,
                        const std::string& extra_info = "");
};

// Global instance declaration
extern MemoryDoctor g_memory_doctor;

#endif // MEMORY_DOCTOR_CUH

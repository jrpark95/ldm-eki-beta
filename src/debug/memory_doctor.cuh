/**
 * @file memory_doctor.cuh
 * @brief Memory Doctor diagnostic system for IPC communication debugging
 *
 * @details
 * Memory Doctor is a specialized debugging tool for diagnosing inter-process
 * communication (IPC) issues between the C++ LDM forward model and the Python
 * EKI inversion system. It logs all data transfers through POSIX shared memory,
 * including comprehensive statistics, checksums, and sample data.
 *
 * Key capabilities:
 * - Bidirectional IPC logging (C++ → Python and Python → C++)
 * - Checksum validation for data integrity verification
 * - Statistical analysis (min/max/mean, zero/NaN/Inf counts)
 * - Sample data logging (first/last 100 elements)
 * - Iteration tracking for temporal debugging
 * - Automatic log directory cleanup on enable
 * - Zero overhead when disabled
 *
 * @use_cases
 * 1. Data corruption diagnosis: Compare sent vs. received checksums
 * 2. Dimension mismatch detection: Verify row/column counts
 * 3. Numerical stability analysis: Check for NaN/Inf propagation
 * 4. Memory alignment issues: Inspect raw data patterns
 * 5. Iteration-by-iteration tracking: Identify when issues first appear
 *
 * @usage Basic pattern:
 * @code
 *   // Enable at start of simulation
 *   g_memory_doctor.setEnabled(true);
 *
 *   // Log data being sent to Python
 *   g_memory_doctor.logSentData("initial_observations", h_obs_data,
 *                               num_receptors, num_timesteps);
 *
 *   // Log data received from Python
 *   g_memory_doctor.logReceivedData("ensemble_states", h_ensemble_data,
 *                                   num_states, num_ensemble, iteration);
 *
 *   // Check logs in logs/memory_doctor/
 *   // Files: iter###_cpp_sent_*.txt and iter###_cpp_recv_*.txt
 * @endcode
 *
 * @log_format
 * Each log file contains:
 * - Header: Type, direction, dimensions, element count
 * - Statistics: Checksum, min/max/mean, special value counts
 * - Sample data: First 100 and last 100 elements (if total > 200)
 * - Extra info: User-provided context string
 *
 * Example log file: logs/memory_doctor/iter001_cpp_sent_initial_observations.txt
 * @code
 *   === MEMORY DOCTOR: C++ SENT DATA ===
 *   Iteration: 1
 *   Type: initial_observations
 *   Direction: C++ → Python
 *   Dimensions: 3 x 72
 *   Total Elements: 216
 *   Checksum: 0xa3f2c8d1
 *   Min: 0.000000
 *   Max: 1.234567e-08
 *   Mean: 5.432109e-10
 *   Zero Count: 198 (91.67%)
 *   Negative Count: 0
 *   NaN Count: 0
 *   Inf Count: 0
 *
 *   === DATA (first 100 elements, last 100 elements) ===
 *   First 100:
 *   0.000000e+00  0.000000e+00  1.234567e-10  ...
 * @endcode
 *
 * @performance
 * - Overhead per log operation: ~1-5ms (file I/O + statistics)
 * - Log file size: ~10-50KB per data transfer
 * - Typical total log size: 1-10MB per simulation (depends on iteration count)
 * - Zero overhead when disabled (early exit in all functions)
 *
 * @thread_safety
 * Not thread-safe. All logging must be done from a single thread (host-side only).
 *
 * @filesystem
 * Log directory: logs/memory_doctor/
 * File naming: iter{###}_{cpp_sent|cpp_recv}_{data_type}.txt
 * Cleanup: All .txt files removed on setEnabled(true)
 *
 * @warning Memory Doctor generates significant I/O overhead
 * @warning Only use for debugging - disable for production runs
 * @warning Log files can grow large for many iterations
 *
 * @author Juryong Park
 * @date 2025-10-16 (Created during IPC refactoring)
 * @see src/ipc/ldm_eki_writer.cu for integration examples
 * @see src/ipc/ldm_eki_reader.cu for integration examples
 */

#ifndef MEMORY_DOCTOR_CUH
#define MEMORY_DOCTOR_CUH

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdint>
#include "../colors.h"

/**
 * @class MemoryDoctor
 * @brief Diagnostic tool for debugging inter-process communication data transfer
 *
 * @details
 * Memory Doctor mode logs all IPC data transfers between C++ and Python
 * processes, including checksums, statistics, and sample data. Useful for
 * diagnosing communication issues, data corruption, or dimension mismatches.
 *
 * @architecture
 * - Singleton pattern (global instance g_memory_doctor)
 * - Enabled/disabled state machine
 * - File-based logging with automatic cleanup
 * - Statistical analysis using single-pass algorithms
 * - Checksum calculation via XOR + rotate-left
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
    std::string log_dir;         ///< Directory for log files (logs/memory_doctor/)

    /**
     * @brief Calculate simple checksum for data verification
     *
     * @details
     * Computes a 32-bit checksum using XOR and rotate-left operations.
     * Not cryptographically secure, but sufficient for detecting data corruption
     * or dimension mismatches in IPC transfers.
     *
     * Algorithm:
     * 1. Interpret float array as uint32_t array (bit pattern reinterpretation)
     * 2. For each element: XOR with accumulator, then rotate accumulator left
     * 3. Rotation prevents simple XOR cancellation (e.g., [a,b,a,b] ≠ 0)
     *
     * @param[in] data   Float array to checksum
     * @param[in] count  Number of elements
     *
     * @return 32-bit checksum value (printed as hex in logs)
     *
     * @note Uses reinterpret_cast to treat float bits as uint32_t
     * @note Rotate-left: (x << 1) | (x >> 31)
     * @note Not cryptographically secure - for debugging only
     *
     * @complexity O(n) single pass
     */
    uint32_t calculateChecksum(const float* data, size_t count) const;

    /**
     * @brief Clean all previous log files in memory_doctor directory
     *
     * @details
     * Removes all .txt files from logs/memory_doctor/ to prevent confusion
     * between different simulation runs. Called automatically when setEnabled(true).
     *
     * @post All *.txt files in log_dir removed
     *
     * @note Uses POSIX directory iteration (opendir/readdir/closedir)
     * @note Only removes .txt files (preserves subdirectories)
     * @note Silent failure if directory doesn't exist (will be created later)
     *
     * @implementation Uses unlink() system call for file deletion
     */
    void cleanLogDirectory();

public:
    /**
     * @brief Construct Memory Doctor with default settings
     *
     * @post enabled = false
     * @post log_dir = "logs/memory_doctor/"
     */
    MemoryDoctor();

    /**
     * @brief Enable or disable Memory Doctor mode
     *
     * @details
     * When enabling:
     * 1. Creates logs/ directory if it doesn't exist
     * 2. Creates logs/memory_doctor/ directory if it doesn't exist
     * 3. Cleans all previous .txt log files
     * 4. Prints status message to console
     *
     * When disabling:
     * - Simply clears enabled flag (logs remain)
     *
     * @param[in] enable  true to enable, false to disable
     *
     * @post enabled = enable
     * @post If enabling: log directory created and cleaned
     * @post Mode status printed to console
     *
     * @note Uses mkdir() with mode 0755 for directory creation
     * @note Creates logs/ and logs/memory_doctor/ if they don't exist
     *
     * @output When enabling:
     * @code
     *   [MEMORY_DOCTOR] ⚕️  Mode ENABLED - All IPC data will be logged
     *   [MEMORY_DOCTOR] 🧹 Cleaned previous log files from logs/memory_doctor/
     * @endcode
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
     * @details
     * Records complete diagnostic information about data transfer, including:
     * - Transfer metadata (type, direction, dimensions)
     * - Checksum for data integrity verification
     * - Statistical summary (min/max/mean, special value counts)
     * - Sample data (first 100 and last 100 elements)
     *
     * Log file format: logs/memory_doctor/iter{###}_cpp_sent_{data_type}.txt
     *
     * @param[in] data_type    Descriptive name (e.g., "initial_observations")
     * @param[in] data         Float array being sent (host memory)
     * @param[in] rows         Number of rows in matrix
     * @param[in] cols         Number of columns in matrix
     * @param[in] iteration    Current EKI iteration number (-1 for non-iterative)
     * @param[in] extra_info   Additional context information (optional)
     *
     * @pre enabled == true (otherwise early exit)
     * @pre data != nullptr
     * @pre rows > 0, cols > 0
     *
     * @post Log file created (if enabled)
     * @post Console message printed with transfer summary
     *
     * @note Creates file: logs/memory_doctor/iter###_cpp_sent_{data_type}.txt
     * @note Logs first 100 and last 100 elements (if total > 200)
     * @note Calculates min/max/mean, zero count, NaN/Inf counts
     * @note If iteration < 0, uses "###" in filename
     *
     * @performance ~1-2ms overhead per call typical (dominated by file I/O)
     *
     * @output Console:
     * @code
     *   [MEMORY_DOCTOR] 📤 Iteration 1: C++ sent initial_observations (3×72) → Python
     * @endcode
     */
    void logSentData(const std::string& data_type, const float* data,
                     int rows, int cols, int iteration = -1,
                     const std::string& extra_info = "");

    /**
     * @brief Log data received by LDM from Python
     *
     * @details
     * Records complete diagnostic information about received data, enabling
     * comparison with Python's sent data logs. Format matches logSentData()
     * for easy side-by-side comparison.
     *
     * Use case: Compare C++ received data with Python sent data to diagnose:
     * - Data corruption during shared memory transfer
     * - Dimension mismatches (row-major vs column-major)
     * - Endianness issues (rare on x86-64)
     * - Memory alignment problems
     *
     * Log file format: logs/memory_doctor/iter{###}_cpp_recv_{data_type}.txt
     *
     * @param[in] data_type    Descriptive name (e.g., "ensemble_states")
     * @param[in] data         Float array received (host memory)
     * @param[in] rows         Number of rows in matrix
     * @param[in] cols         Number of columns in matrix
     * @param[in] iteration    Current EKI iteration number (-1 for non-iterative)
     * @param[in] extra_info   Additional context information (optional)
     *
     * @pre enabled == true (otherwise early exit)
     * @pre data != nullptr
     * @pre rows > 0, cols > 0
     *
     * @post Log file created (if enabled)
     * @post Console message printed with transfer summary
     *
     * @note Creates file: logs/memory_doctor/iter###_cpp_recv_{data_type}.txt
     * @note Format matches logSentData() for easy comparison
     * @note Direction arrow: Python → C++
     *
     * @output Console:
     * @code
     *   [MEMORY_DOCTOR] 📥 Iteration 1: C++ received ensemble_states (7×100) ← Python
     * @endcode
     *
     * @see logSentData() for detailed logging format
     */
    void logReceivedData(const std::string& data_type, const float* data,
                        int rows, int cols, int iteration = -1,
                        const std::string& extra_info = "");
};

/**
 * @var g_memory_doctor
 * @brief Global Memory Doctor instance
 *
 * @details
 * Singleton instance used throughout the codebase for IPC logging.
 * Defined in memory_doctor.cu, declared extern here.
 *
 * @usage
 * @code
 *   extern MemoryDoctor g_memory_doctor;  // In header
 *   g_memory_doctor.setEnabled(true);     // In code
 * @endcode
 */
extern MemoryDoctor g_memory_doctor;

#endif // MEMORY_DOCTOR_CUH

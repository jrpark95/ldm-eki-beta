/**
 * @file kernel_error_collector.cuh
 * @brief Asynchronous CUDA kernel error collection and batch reporting system
 *
 * @details
 * This module provides a centralized error collection mechanism for CUDA kernel
 * errors. Instead of immediately printing errors (which pollutes terminal output),
 * errors are silently collected in memory and reported in batch at simulation end.
 *
 * Key features:
 * - Asynchronous error collection via cudaGetLastError()
 * - Automatic deduplication (same error at same location = single entry)
 * - Frequency counting (how many times each error occurred)
 * - Sorted reporting (most frequent errors first)
 * - Timestamped log file generation
 * - Colorized terminal output for visibility
 * - Zero overhead when no errors occur
 *
 * @note Only collects kernel launch errors and runtime kernel errors
 * @note Does NOT collect synchronous CUDA API errors (cudaMemcpy, etc.)
 * @note Log files only created when errors detected
 *
 * @usage Basic pattern:
 * @code
 *   KernelErrorCollector::enableCollection();    // Start of simulation
 *
 *   myKernel<<<blocks, threads>>>();
 *   CHECK_KERNEL_ERROR();                        // Check after launch
 *
 *   cudaDeviceSynchronize();
 *   CHECK_KERNEL_ERROR();                        // Check after sync
 *
 *   KernelErrorCollector::reportAllErrors();     // End of simulation
 *   KernelErrorCollector::disableCollection();
 * @endcode
 *
 * @architecture
 * Collection Phase:
 *   1. CHECK_KERNEL_ERROR() calls cudaGetLastError()
 *   2. If error found, collectError() stores in vector
 *   3. Duplicate check: same file+line+message → increment count
 *   4. Terminal remains clean (no immediate output)
 *
 * Reporting Phase:
 *   1. Sort errors by frequency (most common first)
 *   2. Print colorized summary to stderr
 *   3. Save to logs/error/kernel_errors_TIMESTAMP.log
 *   4. Clear collection buffer
 *
 * @performance
 * - Collection overhead: ~1 microsecond per check (negligible)
 * - Reporting overhead: ~10ms total (only at simulation end)
 * - Memory usage: ~100 bytes per unique error
 *
 * @author Juryong Park
 * @date 2025-10-16 (Created during output system modernization)
 * @see docs/KERNEL_ERROR_COLLECTOR.md for detailed documentation
 */

#ifndef KERNEL_ERROR_COLLECTOR_CUH
#define KERNEL_ERROR_COLLECTOR_CUH

#include <string>
#include <vector>
#include <map>

namespace KernelErrorCollector {

/**
 * @struct ErrorInfo
 * @brief Storage for a single unique kernel error occurrence
 *
 * @details
 * Represents one unique error location. Multiple occurrences of the same
 * error at the same location increment the count rather than creating
 * duplicate entries.
 *
 * @note Uniqueness determined by: message + file + line
 * @note count starts at 1 in constructor
 */
struct ErrorInfo {
    std::string message;  ///< CUDA error message (e.g., "invalid argument")
    std::string file;     ///< Source filename (basename only, no path)
    int line;             ///< Line number in source file
    int count;            ///< Number of times this error occurred

    /**
     * @brief Construct error info with initial count of 1
     *
     * @param[in] msg  CUDA error message string
     * @param[in] f    Source filename
     * @param[in] l    Line number
     */
    ErrorInfo(const std::string& msg, const std::string& f, int l)
        : message(msg), file(f), line(l), count(1) {}
};

// ============================================================================
// Global State
// ============================================================================

/// Global error collection buffer (vector for dynamic growth)
extern std::vector<ErrorInfo> collected_errors;

/// Whether error collection is currently active
extern bool collection_enabled;

// ============================================================================
// Control Functions
// ============================================================================

/**
 * @brief Enable error collection and clear previous errors
 *
 * @details
 * Activates the error collection system. Should be called at the start of
 * a simulation or test run.
 *
 * @post collection_enabled = true
 * @post collected_errors.clear()
 *
 * @note Safe to call multiple times
 * @note Does not create log directory (that happens during reportAllErrors)
 */
void enableCollection();

/**
 * @brief Disable error collection
 *
 * @details
 * Deactivates the error collection system. Errors will no longer be recorded.
 * Does not clear the buffer - use clearErrors() for that.
 *
 * @post collection_enabled = false
 *
 * @note Does not affect collected_errors buffer
 */
void disableCollection();

/**
 * @brief Clear the error collection buffer
 *
 * @details
 * Removes all collected errors from memory. Typically called after reporting
 * to prepare for next simulation run.
 *
 * @post collected_errors.empty() == true
 */
void clearErrors();

// ============================================================================
// Collection Functions
// ============================================================================

/**
 * @brief Collect a kernel error for batch reporting
 *
 * @details
 * Stores error information in memory for later reporting. If this exact error
 * (same message, file, line) has been seen before, only the count is incremented.
 * Otherwise, a new ErrorInfo entry is created.
 *
 * @param[in] error  CUDA error code from cudaGetLastError()
 * @param[in] file   Source filename (__FILE__)
 * @param[in] line   Line number (__LINE__)
 *
 * @pre collection_enabled == true (otherwise no-op)
 * @pre error != cudaSuccess (otherwise no-op)
 *
 * @post If new error: collected_errors.size() increases by 1
 * @post If duplicate: corresponding ErrorInfo.count increases by 1
 *
 * @note Extracts basename from file path (removes directory components)
 * @note Thread-safe if called from host code only
 *
 * @performance ~1 microsecond per call (vector search + potential insert)
 */
void collectError(cudaError_t error, const char* file, int line);

// ============================================================================
// Reporting Functions
// ============================================================================

/**
 * @brief Report all collected errors to console and log file
 *
 * @details
 * Performs batch error reporting:
 * 1. Sorts errors by frequency (most common first)
 * 2. Prints colorized summary to stderr:
 *    - Red/bold header with separator lines
 *    - Unique error count and total occurrence count
 *    - Each error with location, message, and count
 * 3. Calls saveToFile() to create timestamped log
 *
 * @pre collected_errors may be empty (reports nothing)
 * @post Console displays error report (if errors exist)
 * @post Log file created (if errors exist)
 * @post collected_errors NOT cleared (call clearErrors() if needed)
 *
 * @output Console (stderr):
 * @code
 *   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 *     KERNEL ERROR REPORT
 *   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 *
 *   Total unique errors: 2
 *   Total error occurrences: 648
 *
 *   [1] invalid argument
 *       Location: ldm_func_simulation.cu:392
 *       Count: 432 occurrence(s)
 *
 *   [2] illegal memory access
 *       Location: ldm_func_output.cu:213
 *       Count: 216 occurrence(s)
 *   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * @endcode
 *
 * @see saveToFile() for log file format
 */
void reportAllErrors();

/**
 * @brief Save collected errors to timestamped log file
 *
 * @details
 * Creates a permanent record of kernel errors in the logs/error/ directory.
 * Filename includes timestamp for unique identification across runs.
 *
 * Format: logs/error/kernel_errors_YYYY-MM-DD_HH-MM-SS.log
 *
 * @pre collected_errors may be empty (creates no file)
 * @post Log file created in logs/error/ (if errors exist)
 * @post Success message printed to console
 *
 * @note Creates logs/error/ directory if it doesn't exist
 * @note File format matches console output but without color codes
 * @note Safe to call multiple times (creates new file each time)
 *
 * @output Console (stdout):
 * @code
 *   ✓ Kernel errors saved to logs/error/kernel_errors_2025-10-16_14-32-05.log
 * @endcode
 *
 * @output Log file format:
 * @code
 *   KERNEL ERROR REPORT
 *   Generated: 2025-10-16_14-32-05
 *   ==========================================
 *
 *   Total unique errors: 2
 *   Total error occurrences: 648
 *
 *   [1] invalid argument
 *       Location: ldm_func_simulation.cu:392
 *       Count: 432 occurrence(s)
 *   ...
 * @endcode
 */
void saveToFile();

// ============================================================================
// Convenience Macro
// ============================================================================

/**
 * @def CHECK_KERNEL_ERROR()
 * @brief Macro to check for kernel errors at call site
 *
 * @details
 * Convenience macro that:
 * 1. Calls cudaGetLastError() to retrieve last kernel error
 * 2. If error found, calls collectError() with current location
 * 3. Automatically provides __FILE__ and __LINE__
 *
 * @usage Typical placement:
 * @code
 *   // After kernel launch (catches launch errors)
 *   myKernel<<<blocks, threads>>>(args);
 *   CHECK_KERNEL_ERROR();
 *
 *   // After device sync (catches execution errors)
 *   cudaDeviceSynchronize();
 *   CHECK_KERNEL_ERROR();
 * @endcode
 *
 * @note cudaGetLastError() is destructive (clears error flag)
 * @note Safe to call multiple times (only collects if enabled)
 * @note Zero runtime cost if collection disabled
 *
 * @performance ~1 microsecond overhead per invocation
 */
#define CHECK_KERNEL_ERROR() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            KernelErrorCollector::collectError(err, __FILE__, __LINE__); \
        } \
    } while(0)

} // namespace KernelErrorCollector

#endif // KERNEL_ERROR_COLLECTOR_CUH

/**
 * @file kernel_error_collector.cu
 * @brief Implementation of asynchronous CUDA kernel error collection system
 *
 * @details
 * This module implements the kernel error collection system that allows silent
 * gathering of CUDA kernel errors during simulation, followed by batch reporting
 * at the end. This approach keeps the terminal output clean during normal operation
 * while still providing comprehensive error diagnostics when problems occur.
 *
 * @architecture
 * The system uses a simple vector-based storage with linear search for duplicate
 * detection. This is efficient enough for typical error counts (< 100 unique errors)
 * and avoids the overhead of maintaining a hash map.
 *
 * @thread_safety
 * Not thread-safe. Designed for single-threaded host-side error collection only.
 * Do not call collection functions from multiple threads concurrently.
 *
 * @author Juryong Park
 * @date 2025-10-16 (Created during output system modernization)
 */

#include "kernel_error_collector.cuh"
#include "../colors.h"
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <fstream>

namespace KernelErrorCollector {

// ============================================================================
// Global State Definitions
// ============================================================================

/// Global error collection buffer (defined here, declared in header)
std::vector<ErrorInfo> collected_errors;

/// Whether error collection is currently active (defined here, declared in header)
bool collection_enabled = false;

// ============================================================================
// Control Functions
// ============================================================================

/**
 * @brief Enable error collection and clear previous errors
 *
 * @implementation
 * Sets the enabled flag and clears the error vector. No directory creation
 * or file I/O is performed at this stage (deferred until reporting).
 */
void enableCollection() {
    collection_enabled = true;
    collected_errors.clear();
}

/**
 * @brief Disable error collection
 *
 * @implementation
 * Simply clears the enabled flag. The error buffer is left intact so that
 * errors can still be reported after collection is disabled.
 */
void disableCollection() {
    collection_enabled = false;
}

/**
 * @brief Clear the error collection buffer
 *
 * @implementation
 * Clears the vector, freeing all memory associated with stored errors.
 */
void clearErrors() {
    collected_errors.clear();
}

// ============================================================================
// Collection Functions
// ============================================================================

/**
 * @brief Collect a kernel error for batch reporting
 *
 * @implementation
 * Algorithm:
 * 1. Check if collection is enabled and error is not cudaSuccess (early exit)
 * 2. Extract error message string from CUDA error code
 * 3. Extract basename from file path (remove directory components)
 * 4. Linear search through collected_errors for duplicate
 * 5. If found: increment count
 * 6. If not found: create new ErrorInfo entry
 *
 * @complexity
 * - Time: O(n) where n = number of unique errors (typically < 100)
 * - Space: O(1) if duplicate, O(1) if new (amortized vector growth)
 *
 * @param[in] error  CUDA error code from cudaGetLastError()
 * @param[in] file   Source filename (__FILE__)
 * @param[in] line   Line number (__LINE__)
 */
void collectError(cudaError_t error, const char* file, int line) {
    // Early exit if collection disabled or no error
    if (!collection_enabled || error == cudaSuccess) {
        return;
    }

    // Extract error message from CUDA error code
    std::string error_msg = cudaGetErrorString(error);
    std::string filename = file;

    // Extract basename (remove directory path)
    size_t last_slash = filename.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        filename = filename.substr(last_slash + 1);
    }

    // Check for duplicate error (same message, file, line)
    for (auto& err : collected_errors) {
        if (err.message == error_msg && err.file == filename && err.line == line) {
            err.count++;
            return;  // Duplicate found, increment count and exit
        }
    }

    // New unique error - add to collection
    collected_errors.emplace_back(error_msg, filename, line);
}

// ============================================================================
// Reporting Functions
// ============================================================================

/**
 * @brief Report all collected errors to console and log file
 *
 * @implementation
 * Reporting algorithm:
 * 1. Check if any errors collected (early exit if empty)
 * 2. Sort errors by count (descending) using std::sort + lambda
 * 3. Calculate total error count (sum of all individual counts)
 * 4. Print colorized header with separator lines (red/bold)
 * 5. Print summary statistics (unique count, total count)
 * 6. Iterate through sorted errors, printing each with location
 * 7. Print footer separator
 * 8. Call saveToFile() to create persistent log
 *
 * @output_format
 * Uses ANSI color codes for terminal output:
 * - Color::RED + Color::BOLD: Headers and separators
 * - Color::YELLOW: Statistics labels
 * - Color::CYAN: File locations
 * - Color::RESET: End of colored sections
 */
void reportAllErrors() {
    // Early exit if no errors collected
    if (collected_errors.empty()) {
        return;
    }

    // Sort errors by frequency (most common first)
    std::sort(collected_errors.begin(), collected_errors.end(),
              [](const ErrorInfo& a, const ErrorInfo& b) {
                  return a.count > b.count;
              });

    // Calculate total error occurrences
    int total_errors = 0;
    for (const auto& err : collected_errors) {
        total_errors += err.count;
    }

    // Print colorized header
    std::cerr << "\n";
    std::cerr << Color::RED << Color::BOLD << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cerr << "  KERNEL ERROR REPORT\n";
    std::cerr << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << Color::RESET << "\n\n";

    // Print summary statistics
    std::cerr << Color::YELLOW << "Total unique errors: " << Color::RESET << Color::BOLD
              << collected_errors.size() << Color::RESET << "\n";
    std::cerr << Color::YELLOW << "Total error occurrences: " << Color::RESET << Color::BOLD
              << total_errors << Color::RESET << "\n\n";

    // Print each error with numbering
    for (size_t i = 0; i < collected_errors.size(); i++) {
        const auto& err = collected_errors[i];

        std::cerr << Color::RED << Color::BOLD << "[" << (i+1) << "] " << Color::RESET;
        std::cerr << Color::RED << err.message << Color::RESET << "\n";
        std::cerr << "    Location: " << Color::CYAN << err.file << ":" << err.line << Color::RESET << "\n";
        std::cerr << "    Count: " << Color::YELLOW << err.count << " occurrence(s)" << Color::RESET << "\n";

        // Add blank line between errors (except after last)
        if (i < collected_errors.size() - 1) {
            std::cerr << "\n";
        }
    }

    // Print footer
    std::cerr << "\n";
    std::cerr << Color::RED << Color::BOLD << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << Color::RESET << "\n\n";

    // Save to persistent log file
    saveToFile();
}

/**
 * @brief Save collected errors to timestamped log file
 *
 * @implementation
 * File creation algorithm:
 * 1. Check if any errors collected (early exit if empty)
 * 2. Generate timestamp using localtime() and strftime()
 * 3. Construct filename: logs/error/kernel_errors_TIMESTAMP.log
 * 4. Open file for writing (warning if fails)
 * 5. Write header with timestamp
 * 6. Write summary statistics (unique count, total count)
 * 7. Write each error with location and count
 * 8. Close file
 * 9. Print success message to console
 *
 * @filesystem
 * Assumes logs/error/ directory exists (created by main_eki.cu during startup).
 * If directory doesn't exist, file creation will fail silently with warning.
 *
 * @output_format
 * Plain text format without ANSI color codes (for portability and readability).
 * Format matches console output structure for easy comparison.
 */
void saveToFile() {
    // Early exit if no errors to save
    if (collected_errors.empty()) {
        return;
    }

    // Generate timestamp string
    time_t now = time(nullptr);
    struct tm* timeinfo = localtime(&now);
    char timestamp[100];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d_%H-%M-%S", timeinfo);

    // Construct filename with timestamp
    std::ostringstream filename;
    filename << "logs/error/kernel_errors_" << timestamp << ".log";

    // Open file for writing
    std::ofstream file(filename.str());
    if (!file.is_open()) {
        std::cerr << Color::YELLOW << "[WARNING] " << Color::RESET
                  << "Could not save kernel errors to " << filename.str() << std::endl;
        return;
    }

    // Calculate total error count
    int total_errors = 0;
    for (const auto& err : collected_errors) {
        total_errors += err.count;
    }

    // Write header
    file << "KERNEL ERROR REPORT\n";
    file << "Generated: " << timestamp << "\n";
    file << "==========================================\n\n";

    // Write summary statistics
    file << "Total unique errors: " << collected_errors.size() << "\n";
    file << "Total error occurrences: " << total_errors << "\n\n";

    // Write each error with numbering
    for (size_t i = 0; i < collected_errors.size(); i++) {
        const auto& err = collected_errors[i];

        file << "[" << (i+1) << "] " << err.message << "\n";
        file << "    Location: " << err.file << ":" << err.line << "\n";
        file << "    Count: " << err.count << " occurrence(s)\n";

        // Add blank line between errors (except after last)
        if (i < collected_errors.size() - 1) {
            file << "\n";
        }
    }

    file.close();

    // Print success message
    std::cerr << Color::GREEN << "✓ " << Color::RESET
              << "Kernel errors saved to " << filename.str() << "\n\n";
}

} // namespace KernelErrorCollector

/**
 * @file memory_doctor.cu
 * @brief Implementation of Memory Doctor IPC debugging system
 *
 * @details
 * This module implements the Memory Doctor diagnostic system for debugging
 * inter-process communication between C++ (LDM) and Python (EKI). It provides
 * comprehensive logging of all shared memory data transfers, including:
 *
 * - Checksum calculation for data integrity verification
 * - Statistical analysis (min/max/mean, special value counts)
 * - Sample data logging (first/last 100 elements)
 * - Automatic log directory management
 * - Iteration tracking for temporal debugging
 *
 * @architecture
 * The system uses file-based logging with automatic cleanup. Each data transfer
 * creates a separate log file with a standardized naming convention that includes
 * iteration number, direction, and data type.
 *
 * File naming: logs/memory_doctor/iter{###}_{cpp_sent|cpp_recv}_{data_type}.txt
 *
 * @thread_safety
 * Not thread-safe. All logging operations must be called from a single thread
 * (typically the main host thread).
 *
 * @author Juryong Park
 * @date 2025-10-16 (Created during IPC refactoring)
 */

#include "memory_doctor.cuh"
#include <iomanip>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>

// ============================================================================
// Global Instance Definition
// ============================================================================

/// Global Memory Doctor instance (defined here, declared extern in header)
MemoryDoctor g_memory_doctor;

// ============================================================================
// Constructor
// ============================================================================

/**
 * @brief Construct Memory Doctor with default settings
 *
 * @implementation
 * Initializes the object with disabled state and default log directory path.
 * No file system operations are performed during construction.
 */
MemoryDoctor::MemoryDoctor()
    : enabled(false), log_dir("logs/memory_doctor/") {
}

// ============================================================================
// Control Functions
// ============================================================================

/**
 * @brief Enable or disable Memory Doctor mode
 *
 * @implementation
 * Enable sequence:
 * 1. Create logs/ directory with mkdir() (mode 0755)
 * 2. Create logs/memory_doctor/ directory with mkdir() (mode 0755)
 * 3. Call cleanLogDirectory() to remove previous .txt files
 * 4. Print status messages to console
 * 5. Set enabled flag to true
 *
 * Disable sequence:
 * - Simply set enabled flag to false (logs remain intact)
 *
 * @param[in] enable  true to enable, false to disable
 */
void MemoryDoctor::setEnabled(bool enable) {
    enabled = enable;
    if (enabled) {
        // Create log directory hierarchy if it doesn't exist
        mkdir("logs", 0755);
        mkdir(log_dir.c_str(), 0755);

        // Clean all existing log files from previous runs
        cleanLogDirectory();

        std::cout << Color::YELLOW << "[MEMORY_DOCTOR] " << Color::RESET
                  << "⚕️  Mode ENABLED - All IPC data will be logged\n";
        std::cout << Color::YELLOW << "[MEMORY_DOCTOR] " << Color::RESET
                  << "🧹 Cleaned previous log files from " << Color::BOLD << log_dir << Color::RESET << "\n";
    }
}

// ============================================================================
// Directory Management
// ============================================================================

/**
 * @brief Clean all previous log files in memory_doctor directory
 *
 * @implementation
 * Algorithm:
 * 1. Open directory with opendir()
 * 2. Iterate through all entries with readdir()
 * 3. Check if entry is a .txt file (suffix match)
 * 4. If yes: construct full path and unlink() the file
 * 5. Close directory with closedir()
 *
 * @note Silent failure if directory doesn't exist (will be created later)
 * @note Only removes .txt files (preserves subdirectories)
 *
 * @complexity O(n) where n = number of files in directory
 */
void MemoryDoctor::cleanLogDirectory() {
    DIR* dir = opendir(log_dir.c_str());
    if (dir == nullptr) return;  // Directory doesn't exist yet (OK)

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        // Check if filename ends with .txt (length > 4 and last 4 chars == ".txt")
        if (filename.length() > 4 && filename.substr(filename.length() - 4) == ".txt") {
            std::string full_path = log_dir + filename;
            unlink(full_path.c_str());  // Delete file
        }
    }
    closedir(dir);
}

// ============================================================================
// Checksum Calculation
// ============================================================================

/**
 * @brief Calculate simple checksum for data verification
 *
 * @implementation
 * Algorithm:
 * 1. Reinterpret float array as uint32_t array (bit pattern)
 * 2. Initialize accumulator to 0
 * 3. For each element:
 *    a. XOR accumulator with element
 *    b. Rotate accumulator left by 1 bit: (x << 1) | (x >> 31)
 * 4. Return final accumulator value
 *
 * Properties:
 * - Order-dependent (rotation prevents XOR cancellation)
 * - Bit-level comparison (treats NaN/Inf correctly)
 * - Fast (single pass, simple operations)
 * - Not cryptographically secure
 *
 * @param[in] data   Float array to checksum
 * @param[in] count  Number of elements
 *
 * @return 32-bit checksum value
 *
 * @complexity O(n) single pass
 */
uint32_t MemoryDoctor::calculateChecksum(const float* data, size_t count) const {
    uint32_t checksum = 0;
    const uint32_t* uint_data = reinterpret_cast<const uint32_t*>(data);
    for (size_t i = 0; i < count; ++i) {
        checksum ^= uint_data[i];                    // XOR with element
        checksum = (checksum << 1) | (checksum >> 31); // Rotate left 1 bit
    }
    return checksum;
}

// ============================================================================
// Data Logging Functions
// ============================================================================

/**
 * @brief Log data being sent from LDM to Python
 *
 * @implementation
 * Logging algorithm:
 * 1. Early exit if not enabled
 * 2. Construct filename with iteration number and data type
 * 3. Open file for writing
 * 4. Write header with metadata
 * 5. Calculate statistics in single pass:
 *    - Min/max/sum for non-special values
 *    - Count zeros, negatives, NaN, Inf
 * 6. Write statistics with checksum
 * 7. Write first 100 elements (10 per line)
 * 8. Write last 100 elements if total > 200
 * 9. Close file
 * 10. Print console summary
 *
 * @param[in] data_type    Descriptive name
 * @param[in] data         Float array being sent
 * @param[in] rows         Number of rows
 * @param[in] cols         Number of columns
 * @param[in] iteration    Current iteration number
 * @param[in] extra_info   Additional context
 *
 * @complexity O(n) where n = rows * cols (single pass statistics)
 */
void MemoryDoctor::logSentData(const std::string& data_type, const float* data,
                               int rows, int cols, int iteration,
                               const std::string& extra_info) {
    if (!enabled) return;  // Early exit if disabled

    // Construct filename: iter{###}_cpp_sent_{data_type}.txt
    std::stringstream ss;
    ss << log_dir << "iter" << std::setfill('0') << std::setw(3) << iteration
       << "_cpp_sent_" << data_type << ".txt";
    std::string filename = ss.str();

    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "[MEMORY_DOCTOR] Failed to open log file: " << filename << std::endl;
        return;
    }

    // Write header with metadata
    file << "=== MEMORY DOCTOR: C++ SENT DATA ===" << std::endl;
    file << "Iteration: " << iteration << std::endl;
    file << "Type: " << data_type << std::endl;
    file << "Direction: C++ → Python" << std::endl;
    file << "Dimensions: " << rows << " x " << cols << std::endl;
    file << "Total Elements: " << rows * cols << std::endl;

    // Calculate statistics in single pass
    size_t total = rows * cols;
    uint32_t checksum = calculateChecksum(data, total);

    float min_val = data[0], max_val = data[0], sum = 0;
    int zero_count = 0, nan_count = 0, inf_count = 0, neg_count = 0;

    for (size_t i = 0; i < total; ++i) {
        float val = data[i];
        if (std::isnan(val)) nan_count++;
        else if (std::isinf(val)) inf_count++;
        else {
            if (val == 0.0f) zero_count++;
            if (val < 0.0f) neg_count++;
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
            sum += val;
        }
    }

    // Write statistics
    file << "Checksum: 0x" << std::hex << checksum << std::dec << std::endl;
    file << "Min: " << min_val << std::endl;
    file << "Max: " << max_val << std::endl;
    file << "Mean: " << (sum / total) << std::endl;
    file << "Zero Count: " << zero_count << " (" << (100.0f * zero_count / total) << "%)" << std::endl;
    file << "Negative Count: " << neg_count << std::endl;
    file << "NaN Count: " << nan_count << std::endl;
    file << "Inf Count: " << inf_count << std::endl;

    if (!extra_info.empty()) {
        file << "Extra Info: " << extra_info << std::endl;
    }

    file << std::endl << "=== DATA (first 100 elements, last 100 elements) ===" << std::endl;

    // Write first 100 elements (10 per line, scientific notation)
    file << "First 100:" << std::endl;
    size_t first_count = std::min(size_t(100), total);
    for (size_t i = 0; i < first_count; ++i) {
        file << std::setw(12) << std::scientific << data[i];
        if ((i + 1) % 10 == 0) file << std::endl;
        else file << " ";
    }

    // Write last 100 elements if array is large enough
    if (total > 200) {
        file << std::endl << "Last 100:" << std::endl;
        size_t start = total - 100;
        for (size_t i = start; i < total; ++i) {
            file << std::setw(12) << std::scientific << data[i];
            if ((i - start + 1) % 10 == 0) file << std::endl;
            else file << " ";
        }
    }

    file << std::endl << "=== END OF DATA ===" << std::endl;
    file.close();

    // Print console summary
    std::cout << Color::YELLOW << "[MEMORY_DOCTOR] " << Color::RESET
              << "📤 Iteration " << Color::BOLD << iteration << Color::RESET
              << ": C++ sent " << data_type
              << " (" << rows << "×" << cols << ") → Python\n";
}

// ============================================================================
// Received Data Logging
// ============================================================================

/**
 * @brief Log data received by LDM from Python
 *
 * @implementation
 * Identical algorithm to logSentData(), but with:
 * - Different filename prefix: cpp_recv instead of cpp_sent
 * - Different header direction: Python → C++
 * - Different console arrow: ← instead of →
 *
 * This symmetry allows easy comparison of sent and received logs to diagnose
 * IPC issues.
 *
 * @param[in] data_type    Descriptive name
 * @param[in] data         Float array received
 * @param[in] rows         Number of rows
 * @param[in] cols         Number of columns
 * @param[in] iteration    Current iteration number
 * @param[in] extra_info   Additional context
 *
 * @see logSentData() for detailed algorithm description
 */
void MemoryDoctor::logReceivedData(const std::string& data_type, const float* data,
                                  int rows, int cols, int iteration,
                                  const std::string& extra_info) {
    if (!enabled) return;  // Early exit if disabled

    // Construct filename: iter{###}_cpp_recv_{data_type}.txt
    std::stringstream ss;
    ss << log_dir << "iter" << std::setfill('0') << std::setw(3) << iteration
       << "_cpp_recv_" << data_type << ".txt";
    std::string filename = ss.str();

    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "[MEMORY_DOCTOR] Failed to open log file: " << filename << std::endl;
        return;
    }

    // Write header with metadata (direction: Python → C++)
    file << "=== MEMORY DOCTOR: C++ RECEIVED DATA ===" << std::endl;
    file << "Iteration: " << iteration << std::endl;
    file << "Type: " << data_type << std::endl;
    file << "Direction: Python → C++" << std::endl;
    file << "Dimensions: " << rows << " x " << cols << std::endl;
    file << "Total Elements: " << rows * cols << std::endl;

    // Calculate statistics in single pass
    size_t total = rows * cols;
    uint32_t checksum = calculateChecksum(data, total);

    float min_val = data[0], max_val = data[0], sum = 0;
    int zero_count = 0, nan_count = 0, inf_count = 0, neg_count = 0;

    for (size_t i = 0; i < total; ++i) {
        float val = data[i];
        if (std::isnan(val)) nan_count++;
        else if (std::isinf(val)) inf_count++;
        else {
            if (val == 0.0f) zero_count++;
            if (val < 0.0f) neg_count++;
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
            sum += val;
        }
    }

    // Write statistics
    file << "Checksum: 0x" << std::hex << checksum << std::dec << std::endl;
    file << "Min: " << min_val << std::endl;
    file << "Max: " << max_val << std::endl;
    file << "Mean: " << (sum / total) << std::endl;
    file << "Zero Count: " << zero_count << " (" << (100.0f * zero_count / total) << "%)" << std::endl;
    file << "Negative Count: " << neg_count << std::endl;
    file << "NaN Count: " << nan_count << std::endl;
    file << "Inf Count: " << inf_count << std::endl;

    if (!extra_info.empty()) {
        file << "Extra Info: " << extra_info << std::endl;
    }

    file << std::endl << "=== DATA (first 100 elements, last 100 elements) ===" << std::endl;

    // Write first 100 elements (10 per line, scientific notation)
    file << "First 100:" << std::endl;
    size_t first_count = std::min(size_t(100), total);
    for (size_t i = 0; i < first_count; ++i) {
        file << std::setw(12) << std::scientific << data[i];
        if ((i + 1) % 10 == 0) file << std::endl;
        else file << " ";
    }

    // Write last 100 elements if array is large enough
    if (total > 200) {
        file << std::endl << "Last 100:" << std::endl;
        size_t start = total - 100;
        for (size_t i = start; i < total; ++i) {
            file << std::setw(12) << std::scientific << data[i];
            if ((i - start + 1) % 10 == 0) file << std::endl;
            else file << " ";
        }
    }

    file << std::endl << "=== END OF DATA ===" << std::endl;
    file.close();

    // Print console summary (arrow direction: ← from Python)
    std::cout << Color::YELLOW << "[MEMORY_DOCTOR] " << Color::RESET
              << "Iteration " << Color::BOLD << iteration << Color::RESET
              << ": C++ received " << data_type
              << " (" << rows << "×" << cols << ") ← Python\n";
}

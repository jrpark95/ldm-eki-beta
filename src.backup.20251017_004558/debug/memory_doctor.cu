// memory_doctor.cu - Implementation of Memory Doctor debugging tool
#include "memory_doctor.cuh"
#include <iomanip>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>

// Global instance definition
MemoryDoctor g_memory_doctor;

// ============================================================================
// Constructor
// ============================================================================

MemoryDoctor::MemoryDoctor()
    : enabled(false), log_dir("logs/memory_doctor/") {
}

// ============================================================================
// Enable/Disable
// ============================================================================

void MemoryDoctor::setEnabled(bool enable) {
    enabled = enable;
    if (enabled) {
        // Create log directory if it doesn't exist
        mkdir("logs", 0755);
        mkdir(log_dir.c_str(), 0755);

        // Clean all existing log files in memory_doctor folder
        cleanLogDirectory();

        std::cout << Color::YELLOW << "[MEMORY_DOCTOR] " << Color::RESET
                  << "⚕️  Mode ENABLED - All IPC data will be logged\n";
        std::cout << Color::YELLOW << "[MEMORY_DOCTOR] " << Color::RESET
                  << "🧹 Cleaned previous log files from " << Color::BOLD << log_dir << Color::RESET << "\n";
    }
}

// ============================================================================
// Directory Cleaning
// ============================================================================

void MemoryDoctor::cleanLogDirectory() {
    DIR* dir = opendir(log_dir.c_str());
    if (dir == nullptr) return;

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        // Check if it's a .txt file
        if (filename.length() > 4 && filename.substr(filename.length() - 4) == ".txt") {
            std::string full_path = log_dir + filename;
            unlink(full_path.c_str());
        }
    }
    closedir(dir);
}

// ============================================================================
// Checksum Calculation
// ============================================================================

uint32_t MemoryDoctor::calculateChecksum(const float* data, size_t count) const {
    uint32_t checksum = 0;
    const uint32_t* uint_data = reinterpret_cast<const uint32_t*>(data);
    for (size_t i = 0; i < count; ++i) {
        checksum ^= uint_data[i];
        checksum = (checksum << 1) | (checksum >> 31); // Rotate left
    }
    return checksum;
}

// ============================================================================
// Log Sent Data
// ============================================================================

void MemoryDoctor::logSentData(const std::string& data_type, const float* data,
                               int rows, int cols, int iteration,
                               const std::string& extra_info) {
    if (!enabled) return;

    // Format: iter{000}_cpp_sent_{data_type}.txt
    std::stringstream ss;
    ss << log_dir << "iter" << std::setfill('0') << std::setw(3) << iteration
       << "_cpp_sent_" << data_type << ".txt";
    std::string filename = ss.str();

    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "[MEMORY_DOCTOR] Failed to open log file: " << filename << std::endl;
        return;
    }

    // Write header
    file << "=== MEMORY DOCTOR: C++ SENT DATA ===" << std::endl;
    file << "Iteration: " << iteration << std::endl;
    file << "Type: " << data_type << std::endl;
    file << "Direction: C++ → Python" << std::endl;
    file << "Dimensions: " << rows << " x " << cols << std::endl;
    file << "Total Elements: " << rows * cols << std::endl;

    // Calculate statistics
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

    // Write first 100 elements
    file << "First 100:" << std::endl;
    size_t first_count = std::min(size_t(100), total);
    for (size_t i = 0; i < first_count; ++i) {
        file << std::setw(12) << std::scientific << data[i];
        if ((i + 1) % 10 == 0) file << std::endl;
        else file << " ";
    }

    // Write last 100 elements if different from first
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

    std::cout << Color::YELLOW << "[MEMORY_DOCTOR] " << Color::RESET
              << "📤 Iteration " << Color::BOLD << iteration << Color::RESET
              << ": C++ sent " << data_type
              << " (" << rows << "×" << cols << ") → Python\n";
}

// ============================================================================
// Log Received Data
// ============================================================================

void MemoryDoctor::logReceivedData(const std::string& data_type, const float* data,
                                  int rows, int cols, int iteration,
                                  const std::string& extra_info) {
    if (!enabled) return;

    // Format: iter{000}_cpp_recv_{data_type}.txt
    std::stringstream ss;
    ss << log_dir << "iter" << std::setfill('0') << std::setw(3) << iteration
       << "_cpp_recv_" << data_type << ".txt";
    std::string filename = ss.str();

    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "[MEMORY_DOCTOR] Failed to open log file: " << filename << std::endl;
        return;
    }

    // Write header
    file << "=== MEMORY DOCTOR: C++ RECEIVED DATA ===" << std::endl;
    file << "Iteration: " << iteration << std::endl;
    file << "Type: " << data_type << std::endl;
    file << "Direction: Python → C++" << std::endl;
    file << "Dimensions: " << rows << " x " << cols << std::endl;
    file << "Total Elements: " << rows * cols << std::endl;

    // Calculate statistics
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

    // Write first 100 elements
    file << "First 100:" << std::endl;
    size_t first_count = std::min(size_t(100), total);
    for (size_t i = 0; i < first_count; ++i) {
        file << std::setw(12) << std::scientific << data[i];
        if ((i + 1) % 10 == 0) file << std::endl;
        else file << " ";
    }

    // Write last 100 elements if different from first
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

    std::cout << Color::YELLOW << "[MEMORY_DOCTOR] " << Color::RESET
              << "📥 Iteration " << Color::BOLD << iteration << Color::RESET
              << ": C++ received " << data_type
              << " (" << rows << "×" << cols << ") ← Python\n";
}

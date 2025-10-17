/**
 * @file kernel_error_collector.cu
 * @brief Kernel error collection system implementation
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

// Global variables
std::vector<ErrorInfo> collected_errors;
bool collection_enabled = false;

void enableCollection() {
    collection_enabled = true;
    collected_errors.clear();
}

void disableCollection() {
    collection_enabled = false;
}

void collectError(cudaError_t error, const char* file, int line) {
    if (!collection_enabled || error == cudaSuccess) {
        return;
    }

    std::string error_msg = cudaGetErrorString(error);
    std::string filename = file;

    // Extract just the filename from full path
    size_t last_slash = filename.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        filename = filename.substr(last_slash + 1);
    }

    // Check if this error already exists (same message, file, line)
    for (auto& err : collected_errors) {
        if (err.message == error_msg && err.file == filename && err.line == line) {
            err.count++;
            return;
        }
    }

    // New error - add to collection
    collected_errors.emplace_back(error_msg, filename, line);
}

void reportAllErrors() {
    if (collected_errors.empty()) {
        return;
    }

    // Sort errors by count (descending)
    std::sort(collected_errors.begin(), collected_errors.end(),
              [](const ErrorInfo& a, const ErrorInfo& b) {
                  return a.count > b.count;
              });

    // Print header
    std::cerr << "\n";
    std::cerr << Color::RED << Color::BOLD << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cerr << "  KERNEL ERROR REPORT\n";
    std::cerr << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << Color::RESET << "\n\n";

    int total_errors = 0;
    for (const auto& err : collected_errors) {
        total_errors += err.count;
    }

    std::cerr << Color::YELLOW << "Total unique errors: " << Color::RESET << Color::BOLD
              << collected_errors.size() << Color::RESET << "\n";
    std::cerr << Color::YELLOW << "Total error occurrences: " << Color::RESET << Color::BOLD
              << total_errors << Color::RESET << "\n\n";

    // Print each error
    for (size_t i = 0; i < collected_errors.size(); i++) {
        const auto& err = collected_errors[i];

        std::cerr << Color::RED << Color::BOLD << "[" << (i+1) << "] " << Color::RESET;
        std::cerr << Color::RED << err.message << Color::RESET << "\n";
        std::cerr << "    Location: " << Color::CYAN << err.file << ":" << err.line << Color::RESET << "\n";
        std::cerr << "    Count: " << Color::YELLOW << err.count << " occurrence(s)" << Color::RESET << "\n";

        if (i < collected_errors.size() - 1) {
            std::cerr << "\n";
        }
    }

    std::cerr << "\n";
    std::cerr << Color::RED << Color::BOLD << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << Color::RESET << "\n\n";

    // Save to file
    saveToFile();
}

void saveToFile() {
    if (collected_errors.empty()) {
        return;
    }

    // Create timestamp
    time_t now = time(nullptr);
    struct tm* timeinfo = localtime(&now);
    char timestamp[100];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d_%H-%M-%S", timeinfo);

    // Create filename
    std::ostringstream filename;
    filename << "logs/error/kernel_errors_" << timestamp << ".log";

    std::ofstream file(filename.str());
    if (!file.is_open()) {
        std::cerr << Color::YELLOW << "[WARNING] " << Color::RESET
                  << "Could not save kernel errors to " << filename.str() << std::endl;
        return;
    }

    // Write header
    file << "KERNEL ERROR REPORT\n";
    file << "Generated: " << timestamp << "\n";
    file << "==========================================\n\n";

    int total_errors = 0;
    for (const auto& err : collected_errors) {
        total_errors += err.count;
    }

    file << "Total unique errors: " << collected_errors.size() << "\n";
    file << "Total error occurrences: " << total_errors << "\n\n";

    // Write each error
    for (size_t i = 0; i < collected_errors.size(); i++) {
        const auto& err = collected_errors[i];

        file << "[" << (i+1) << "] " << err.message << "\n";
        file << "    Location: " << err.file << ":" << err.line << "\n";
        file << "    Count: " << err.count << " occurrence(s)\n";

        if (i < collected_errors.size() - 1) {
            file << "\n";
        }
    }

    file.close();

    std::cerr << Color::GREEN << "✓ " << Color::RESET
              << "Kernel errors saved to " << filename.str() << "\n\n";
}

void clearErrors() {
    collected_errors.clear();
}

} // namespace KernelErrorCollector

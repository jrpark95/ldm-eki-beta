/**
 * @file kernel_error_collector.cuh
 * @brief Kernel error collection system for batch reporting
 */

#ifndef KERNEL_ERROR_COLLECTOR_CUH
#define KERNEL_ERROR_COLLECTOR_CUH

#include <string>
#include <vector>
#include <map>

namespace KernelErrorCollector {

struct ErrorInfo {
    std::string message;
    std::string file;
    int line;
    int count;

    ErrorInfo(const std::string& msg, const std::string& f, int l)
        : message(msg), file(f), line(l), count(1) {}
};

// Global error collection
extern std::vector<ErrorInfo> collected_errors;
extern bool collection_enabled;

// Enable/disable error collection
void enableCollection();
void disableCollection();

// Collect a kernel error
void collectError(cudaError_t error, const char* file, int line);

// Report all collected errors at once
void reportAllErrors();

// Save errors to timestamped log file
void saveToFile();

// Clear error buffer
void clearErrors();

// Macro for checking kernel errors
#define CHECK_KERNEL_ERROR() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            KernelErrorCollector::collectError(err, __FILE__, __LINE__); \
        } \
    } while(0)

} // namespace KernelErrorCollector

#endif // KERNEL_ERROR_COLLECTOR_CUH

// ldm_eki_reader.cu - Implementation of EKI IPC Reader
#include "ldm_eki_reader.cuh"
#include "../debug/memory_doctor.cuh"
#include <numeric>
#include <algorithm>

namespace LDM_EKI_IPC {

// ============================================================================
// Constructor / Destructor
// ============================================================================

EKIReader::EKIReader()
    : config_fd(-1), data_fd(-1), config_map(nullptr),
      data_map(nullptr), data_size(0), initialized(false) {
}

EKIReader::~EKIReader() {
    cleanup();
}

// ============================================================================
// Wait for Ensemble Data
// ============================================================================

bool EKIReader::waitForEnsembleData(int timeout_seconds, int expected_iteration) {
    std::cout << Color::CYAN << "[IPC] " << Color::RESET
              << "Waiting for ensemble data from Python (timeout: "
              << Color::BOLD << timeout_seconds << "s" << Color::RESET << ")...\n";

    const char* config_path = "/dev/shm/ldm_eki_ensemble_config";
    const char* data_path = "/dev/shm/ldm_eki_ensemble_data";

    // Store last iteration ID to detect fresh data
    static int last_iteration_id = -1;

    // Wait for config file to appear with fresh data
    for (int i = 0; i < timeout_seconds; i++) {
        if (access(config_path, F_OK) == 0 && access(data_path, F_OK) == 0) {
            // Read config to check iteration ID
            int config_fd = open(config_path, O_RDONLY);
            if (config_fd >= 0) {
                EnsembleConfig config;
                ssize_t bytes_read = read(config_fd, &config, sizeof(config));
                close(config_fd);

                if (bytes_read == sizeof(config)) {
                    // Check if this is new data (different iteration ID)
                    if (config.timestep_id > last_iteration_id) {
                        // New iteration detected, now check if data is ready
                        int test_fd = open(data_path, O_RDONLY);
                        if (test_fd >= 0) {
                            EnsembleDataHeader header;
                            bytes_read = read(test_fd, &header, sizeof(header));
                            close(test_fd);

                            if (bytes_read == sizeof(header) && header.status == 1) {
                                std::cout << Color::GREEN << "✓ " << Color::RESET
                                          << "Fresh ensemble data detected (iteration " << Color::BOLD
                                          << config.timestep_id << Color::RESET << ")\n";
                                last_iteration_id = config.timestep_id;
                                return true;
                            }
                        }
                    } else if (config.timestep_id == last_iteration_id && i > 5) {
                        // Same iteration ID after 5 seconds - probably stale data
                        if (i % 5 == 0) {
                            std::cout << "\rWaiting for new data... (iteration " << config.timestep_id << ")" << std::flush;
                        }
                    }
                }
            }
        }
        sleep(1);
    }

    std::cerr << Color::RED << Color::BOLD << "[ERROR] " << Color::RESET
              << "Timeout waiting for ensemble data\n";
    return false;
}

// ============================================================================
// Read Configuration
// ============================================================================

bool EKIReader::readEnsembleConfig(int& num_states, int& num_ensemble, int& timestep_id) {
    const char* shm_path = "/dev/shm/ldm_eki_ensemble_config";

    config_fd = open(shm_path, O_RDONLY);
    if (config_fd < 0) {
        fprintf(stderr, "%s[ERROR]%s ", Color::RED, Color::RESET); perror(" Failed to open config");
        return false;
    }

    EnsembleConfig config;
    ssize_t bytes_read = read(config_fd, &config, sizeof(config));
    close(config_fd);
    config_fd = -1;

    if (bytes_read != sizeof(config)) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Failed to read config (got " << bytes_read << " bytes)" << std::endl;
        return false;
    }

    num_states = config.num_states;
    num_ensemble = config.num_ensemble;
    timestep_id = config.timestep_id;

    std::cout << Color::CYAN << "[IPC] " << Color::RESET
              << "Config loaded: " << Color::BOLD << num_states << "×" << num_ensemble << Color::RESET
              << " (timestep " << timestep_id << ")\n";
    return true;
}

// ============================================================================
// Read Ensemble States
// ============================================================================

bool EKIReader::readEnsembleStates(std::vector<float>& output, int& num_states, int& num_ensemble) {
    // First read config
    int timestep_id;
    if (!readEnsembleConfig(num_states, num_ensemble, timestep_id)) {
        return false;
    }

    const char* shm_path = "/dev/shm/ldm_eki_ensemble_data";

    data_fd = open(shm_path, O_RDONLY);
    if (data_fd < 0) {
        fprintf(stderr, "%s[ERROR]%s ", Color::RED, Color::RESET); perror(" Failed to open data");
        return false;
    }

    // Get file size
    struct stat st;
    if (fstat(data_fd, &st) != 0) {
        fprintf(stderr, "%s[ERROR]%s ", Color::RED, Color::RESET); perror(" fstat failed");
        close(data_fd);
        data_fd = -1;
        return false;
    }

    size_t file_size = st.st_size;
    size_t expected_size = sizeof(EnsembleDataHeader) + num_states * num_ensemble * sizeof(float);

    if (file_size != expected_size) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Size mismatch: file=" << file_size
                  << " bytes, expected=" << expected_size << " bytes" << std::endl;
        close(data_fd);
        data_fd = -1;
        return false;
    }

    // Map entire file
    data_map = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, data_fd, 0);
    if (data_map == MAP_FAILED) {
        fprintf(stderr, "%s[ERROR]%s ", Color::RED, Color::RESET); perror(" mmap failed");
        close(data_fd);
        data_fd = -1;
        return false;
    }

    // Read header
    auto* header = reinterpret_cast<EnsembleDataHeader*>(data_map);

    if (header->status != 1) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Data not ready (status=" << header->status << ")" << std::endl;
        munmap(data_map, file_size);
        close(data_fd);
        data_map = nullptr;
        data_fd = -1;
        return false;
    }

    if (header->rows != num_states || header->cols != num_ensemble) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Dimension mismatch: header says " << header->rows
                  << "×" << header->cols << ", config says " << num_states << "×" << num_ensemble << std::endl;
        munmap(data_map, file_size);
        close(data_fd);
        data_map = nullptr;
        data_fd = -1;
        return false;
    }

    // Read data
    float* data_ptr = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(data_map) + sizeof(EnsembleDataHeader)
    );

    size_t data_count = num_states * num_ensemble;
    output.resize(data_count);
    std::memcpy(output.data(), data_ptr, data_count * sizeof(float));

    // Calculate statistics
    float min_val = *std::min_element(output.begin(), output.end());
    float max_val = *std::max_element(output.begin(), output.end());
    float sum = std::accumulate(output.begin(), output.end(), 0.0f);
    float mean_val = sum / data_count;

    std::cout << Color::CYAN << "[IPC] " << Color::RESET
              << "Ensemble states loaded: " << Color::BOLD << num_states << "×" << num_ensemble << Color::RESET
              << " (" << data_count * sizeof(float) / 1024.0 << " KB)\n";
    std::cout << "  Range : [" << min_val << ", " << max_val << "], mean=" << mean_val << "\n";

    // Memory Doctor: Log received ensemble states with iteration from timestep_id
    if (g_memory_doctor.isEnabled()) {
        std::string info = "EKI iteration " + std::to_string(timestep_id) + " from Python";
        g_memory_doctor.logReceivedData("ensemble_states", output.data(),
                                      num_states, num_ensemble, timestep_id, info);
    }

    // Cleanup mapping
    munmap(data_map, file_size);
    close(data_fd);
    data_map = nullptr;
    data_fd = -1;

    return true;
}

// ============================================================================
// Cleanup
// ============================================================================

void EKIReader::cleanup() {
    if (data_map) {
        munmap(data_map, data_size);
        data_map = nullptr;
    }
    if (config_map) {
        munmap(config_map, sizeof(EnsembleConfig));
        config_map = nullptr;
    }
    if (data_fd >= 0) {
        close(data_fd);
        data_fd = -1;
    }
    if (config_fd >= 0) {
        close(config_fd);
        config_fd = -1;
    }
    initialized = false;
}

void EKIReader::unlinkEnsembleSharedMemory() {
    shm_unlink(SHM_ENSEMBLE_CONFIG_NAME);
    shm_unlink(SHM_ENSEMBLE_DATA_NAME);
    std::cout << Color::CYAN << "[IPC] " << Color::RESET
              << "Ensemble shared memory unlinked\n";
}

} // namespace LDM_EKI_IPC

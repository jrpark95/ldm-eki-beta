////////////////////////////////////////////////////////////////////////////////
/// @file    ldm_eki_writer.cu
/// @brief   Implementation of IPC writer for LDM→Python communication
/// @details Implements POSIX shared memory operations for transmitting
///          observation data from C++/CUDA forward model to Python EKI
///          inversion process.
///
/// @author  Juryong Park
/// @date    2025
////////////////////////////////////////////////////////////////////////////////

#include "ldm_eki_writer.cuh"
#include "../core/ldm.cuh"  // For EKIConfig definition
#include "../debug/memory_doctor.cuh"
#include <errno.h>

namespace LDM_EKI_IPC {

////////////////////////////////////////////////////////////////////////////////
// Constructor / Destructor
////////////////////////////////////////////////////////////////////////////////

EKIWriter::EKIWriter()
    : config_fd(-1), data_fd(-1), config_map(nullptr),
      data_map(nullptr), data_size(0), initialized(false) {
}

EKIWriter::~EKIWriter() {
    cleanup();
}

////////////////////////////////////////////////////////////////////////////////
// Initialization
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// @brief Initialize shared memory segments with full configuration
///
/// @details
/// Creates two shared memory segments:
/// 1. Configuration segment (128 bytes) - Contains all EKI parameters
/// 2. Data segment (variable size) - Contains observation data
///
/// The configuration segment includes:
/// - Basic dimensions (ensemble size, receptors, timesteps)
/// - Algorithm parameters (iteration, lambda, noise, etc.)
/// - Option flags (adaptive, localized, regularization, etc.)
/// - GPU settings
/// - Memory Doctor mode
///
/// @param[in] eki_config     Complete EKI configuration structure
/// @param[in] num_timesteps  Number of simulation timesteps
///
/// @return true if initialization successful, false on error
///
/// @post /dev/shm/ldm_eki_config created and written
/// @post /dev/shm/ldm_eki_data created and header initialized
////////////////////////////////////////////////////////////////////////////////
bool EKIWriter::initialize(const ::EKIConfig& eki_config, int num_timesteps) {
    if (initialized) {
        std::cerr << "EKIWriter already initialized" << std::endl;
        return false;
    }

    // Calculate data size: header + receptor data
    data_size = sizeof(EKIDataHeader) + eki_config.num_receptors * num_timesteps * sizeof(float);

    // Create config shared memory (now using EKIConfigFull size)
    config_fd = shm_open(SHM_CONFIG_NAME, O_CREAT | O_RDWR, 0660);
    if (config_fd < 0) {
        perror("shm_open config");
        return false;
    }

    if (ftruncate(config_fd, sizeof(EKIConfigFull)) != 0) {
        perror("ftruncate config");
        close(config_fd);
        return false;
    }

    config_map = mmap(nullptr, sizeof(EKIConfigFull), PROT_READ | PROT_WRITE,
                     MAP_SHARED, config_fd, 0);
    if (config_map == MAP_FAILED) {
        perror("mmap config");
        close(config_fd);
        return false;
    }

    // Write full configuration
    auto* config = reinterpret_cast<EKIConfigFull*>(config_map);
    memset(config, 0, sizeof(EKIConfigFull));

    // Basic info
    config->ensemble_size = eki_config.ensemble_size;
    config->num_receptors = eki_config.num_receptors;
    config->num_timesteps = num_timesteps;

    // Algorithm parameters
    config->iteration = eki_config.iteration;
    config->renkf_lambda = eki_config.renkf_lambda;
    config->noise_level = eki_config.noise_level;
    config->time_interval = eki_config.time_interval;
    config->prior_constant = eki_config.prior_constant;

    // Option strings (safe copy with null termination)
    memset(config->perturb_option, 0, 8);
    strncpy(config->perturb_option, eki_config.perturb_option.c_str(), 7);

    memset(config->adaptive_eki, 0, 8);
    strncpy(config->adaptive_eki, eki_config.adaptive_eki.c_str(), 7);

    memset(config->localized_eki, 0, 8);
    strncpy(config->localized_eki, eki_config.localized_eki.c_str(), 7);

    memset(config->regularization, 0, 8);
    strncpy(config->regularization, eki_config.regularization.c_str(), 7);

    memset(config->time_unit, 0, 8);
    strncpy(config->time_unit, eki_config.time_unit.c_str(), 7);

    memset(config->memory_doctor, 0, 8);
    strncpy(config->memory_doctor, eki_config.memory_doctor_mode ? "On" : "Off", 7);

    // Create data shared memory
    data_fd = shm_open(SHM_DATA_NAME, O_CREAT | O_RDWR, 0660);
    if (data_fd < 0) {
        perror("shm_open data");
        munmap(config_map, sizeof(EKIConfigFull));
        close(config_fd);
        return false;
    }

    if (ftruncate(data_fd, data_size) != 0) {
        perror("ftruncate data");
        munmap(config_map, sizeof(EKIConfigFull));
        close(config_fd);
        close(data_fd);
        return false;
    }

    data_map = mmap(nullptr, data_size, PROT_READ | PROT_WRITE,
                   MAP_SHARED, data_fd, 0);
    if (data_map == MAP_FAILED) {
        perror("mmap data");
        munmap(config_map, sizeof(EKIConfigFull));
        close(config_fd);
        close(data_fd);
        return false;
    }

    // Initialize data header
    auto* header = reinterpret_cast<EKIDataHeader*>(data_map);
    header->status = 0;  // Writing status
    header->rows = eki_config.num_receptors;
    header->cols = num_timesteps;

    initialized = true;
    std::cout << Color::CYAN << "[IPC] " << Color::RESET
              << "Writer initialized with full configuration\n";
    std::cout << "  Ensembles      : " << Color::BOLD << eki_config.ensemble_size << Color::RESET << "\n";
    std::cout << "  Receptors      : " << Color::BOLD << eki_config.num_receptors << Color::RESET << "\n";
    std::cout << "  Timesteps      : " << Color::BOLD << num_timesteps << Color::RESET << "\n";
    std::cout << "  Iteration      : " << eki_config.iteration << "\n";
    std::cout << "  Regularization : " << eki_config.regularization << "\n";
    return true;
}

////////////////////////////////////////////////////////////////////////////////
// Write Initial Observations
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// @brief Write initial observation matrix to shared memory
///
/// @details
/// Writes the "truth" simulation observations to the data segment.
/// Uses a simple handshake protocol:
/// - status=0: Writing in progress
/// - memcpy data
/// - status=1: Ready for Python to read
///
/// @param[in] observations  Row-major observation matrix [receptors × timesteps]
/// @param[in] rows          Number of receptors
/// @param[in] cols          Number of timesteps
///
/// @return true if write successful, false on error
///
/// @pre initialize() must have been called
/// @post Data written to /dev/shm/ldm_eki_data with status=1
////////////////////////////////////////////////////////////////////////////////
bool EKIWriter::writeObservations(const float* observations, int rows, int cols) {
    if (!initialized) {
        std::cerr << "EKIWriter not initialized" << std::endl;
        return false;
    }

    auto* header = reinterpret_cast<EKIDataHeader*>(data_map);

    // Verify dimensions
    if (rows != header->rows || cols != header->cols) {
        std::cerr << "Dimension mismatch: expected " << header->rows
                  << "x" << header->cols << ", got " << rows << "x" << cols << std::endl;
        return false;
    }

    // Set writing status
    header->status = 0;

    // Copy data
    float* data_ptr = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(data_map) + sizeof(EKIDataHeader)
    );

    memcpy(data_ptr, observations, rows * cols * sizeof(float));

    // Memory Doctor: Log sent data (iteration 0 for initial observations)
    if (g_memory_doctor.isEnabled()) {
        g_memory_doctor.logSentData("initial_observations", observations, rows, cols, 0,
                                   "LDM->Python initial EKI observations");
    }

    // Set ready status
    header->status = 1;

    std::cout << Color::CYAN << "[IPC] " << Color::RESET
              << "Observations written: " << Color::BOLD << rows << "×" << cols << Color::RESET
              << " matrix (" << (rows * cols * sizeof(float)) / 1024.0 << " KB)\n";
    return true;
}

////////////////////////////////////////////////////////////////////////////////
// Ensemble Observations
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// @brief Initialize ensemble observation shared memory segments
///
/// @details
/// Creates the configuration segment for ensemble observations.
/// The data segment will be created/truncated on each write.
///
/// @param[in] ensemble_size   Number of ensemble members
/// @param[in] num_receptors   Number of receptors
/// @param[in] num_timesteps   Number of timesteps
///
/// @return true if successful, false on error
///
/// @post /dev/shm/ldm_eki_ensemble_obs_config created and written
////////////////////////////////////////////////////////////////////////////////
bool EKIWriter::initializeEnsembleObservations(int ensemble_size, int num_receptors, int num_timesteps) {
    if (!initialized) {
        std::cerr << "EKIWriter not initialized. Call initialize() first." << std::endl;
        return false;
    }

    // Create ensemble observation config shared memory
    int ens_obs_config_fd = shm_open(SHM_ENSEMBLE_OBS_CONFIG_NAME, O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (ens_obs_config_fd < 0) {
        std::cerr << "Failed to create ensemble obs config shared memory: " << strerror(errno) << std::endl;
        return false;
    }

    // Set size for config (12 bytes)
    if (ftruncate(ens_obs_config_fd, sizeof(EKIConfigBasic)) != 0) {
        std::cerr << "Failed to set ensemble obs config size: " << strerror(errno) << std::endl;
        close(ens_obs_config_fd);
        return false;
    }

    // Map config memory
    void* ens_obs_config_map = mmap(nullptr, sizeof(EKIConfigBasic), PROT_READ | PROT_WRITE, MAP_SHARED, ens_obs_config_fd, 0);
    if (ens_obs_config_map == MAP_FAILED) {
        std::cerr << "Failed to map ensemble obs config memory: " << strerror(errno) << std::endl;
        close(ens_obs_config_fd);
        return false;
    }

    // Write config
    EKIConfigBasic* ens_obs_config = static_cast<EKIConfigBasic*>(ens_obs_config_map);
    ens_obs_config->ensemble_size = ensemble_size;
    ens_obs_config->num_receptors = num_receptors;
    ens_obs_config->num_timesteps = num_timesteps;

    // Unmap config (data will be written separately)
    munmap(ens_obs_config_map, sizeof(EKIConfigBasic));
    close(ens_obs_config_fd);

    std::cout << Color::CYAN << "[IPC] " << Color::RESET
              << "Ensemble observation config: "
              << Color::BOLD << ensemble_size << Color::RESET << " ensembles, "
              << Color::BOLD << num_receptors << Color::RESET << " receptors, "
              << Color::BOLD << num_timesteps << Color::RESET << " timesteps\n";

    return true;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Write ensemble observation tensor to shared memory
///
/// @details
/// Writes all ensemble member observations for the current iteration.
/// Creates/truncates the data segment each time to ensure fresh data.
///
/// Data Layout:
/// - [ensemble_0][receptor_0][time_0], [ensemble_0][receptor_0][time_1], ...
/// - [ensemble_0][receptor_1][time_0], [ensemble_0][receptor_1][time_1], ...
/// - ...
/// - [ensemble_N][receptor_R][time_T]
///
/// @param[in] observations    Flattened 3D tensor (row-major)
/// @param[in] ensemble_size   Number of ensemble members
/// @param[in] num_receptors   Number of receptors
/// @param[in] num_timesteps   Number of timesteps
/// @param[in] iteration       Current EKI iteration (for logging)
///
/// @return true if write successful, false on error
///
/// @pre initializeEnsembleObservations() must have been called
/// @post /dev/shm/ldm_eki_ensemble_obs_data created/truncated and written
////////////////////////////////////////////////////////////////////////////////
bool EKIWriter::writeEnsembleObservations(const float* observations, int ensemble_size,
                                         int num_receptors, int num_timesteps, int iteration) {
    if (!initialized) {
        std::cerr << "EKIWriter not initialized" << std::endl;
        return false;
    }

    // Calculate data size
    size_t ens_obs_data_size = ensemble_size * num_receptors * num_timesteps * sizeof(float);

    // Create/open data shared memory
    int ens_obs_data_fd = shm_open(SHM_ENSEMBLE_OBS_DATA_NAME, O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (ens_obs_data_fd < 0) {
        std::cerr << "Failed to create ensemble obs data shared memory: " << strerror(errno) << std::endl;
        return false;
    }

    // Set data size
    if (ftruncate(ens_obs_data_fd, ens_obs_data_size) != 0) {
        std::cerr << "Failed to set ensemble obs data size: " << strerror(errno) << std::endl;
        close(ens_obs_data_fd);
        return false;
    }

    // Map data memory
    void* ens_obs_data_map = mmap(nullptr, ens_obs_data_size, PROT_READ | PROT_WRITE, MAP_SHARED, ens_obs_data_fd, 0);
    if (ens_obs_data_map == MAP_FAILED) {
        std::cerr << "Failed to map ensemble obs data memory: " << strerror(errno) << std::endl;
        close(ens_obs_data_fd);
        return false;
    }

    // Write data
    memcpy(ens_obs_data_map, observations, ens_obs_data_size);

    // Calculate statistics for validation
    float min_val = observations[0];
    float max_val = observations[0];
    float sum_val = 0.0f;
    int total_elements = ensemble_size * num_receptors * num_timesteps;

    for (int i = 0; i < total_elements; i++) {
        float val = observations[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum_val += val;
    }

    std::cout << Color::CYAN << "[IPC] " << Color::RESET
              << "Ensemble observations written (" << Color::BOLD << ens_obs_data_size / 1024.0 << " KB" << Color::RESET << ")\n";
    std::cout << "  Shape : [" << ensemble_size << " × " << num_receptors << " × " << num_timesteps << "]\n";
    std::cout << "  Range : [" << min_val << ", " << max_val << "], mean=" << (sum_val / total_elements) << "\n";

    // Memory Doctor: Log sent ensemble observations with iteration
    if (g_memory_doctor.isEnabled()) {
        std::string info = "EKI iteration " + std::to_string(iteration) + ": " + std::to_string(ensemble_size) + " ensembles";
        g_memory_doctor.logSentData("ensemble_observations", observations,
                                   ensemble_size * num_receptors, num_timesteps, iteration, info);
    }

    // Unmap data
    munmap(ens_obs_data_map, ens_obs_data_size);
    close(ens_obs_data_fd);

    return true;
}

////////////////////////////////////////////////////////////////////////////////
// Configuration Retrieval
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// @brief Get current basic configuration dimensions
///
/// @param[out] ensemble_size   Number of ensemble members
/// @param[out] num_receptors   Number of receptors
/// @param[out] num_timesteps   Number of timesteps
///
/// @return true if config available, false if not initialized
////////////////////////////////////////////////////////////////////////////////
bool EKIWriter::getConfig(int& ensemble_size, int& num_receptors, int& num_timesteps) {
    if (!initialized) {
        return false;
    }

    auto* config = reinterpret_cast<EKIConfigFull*>(config_map);
    ensemble_size = config->ensemble_size;
    num_receptors = config->num_receptors;
    num_timesteps = config->num_timesteps;
    return true;
}

////////////////////////////////////////////////////////////////////////////////
// Cleanup
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// @brief Cleanup resources and unmap memory
///
/// @details
/// Unmaps memory and closes file descriptors, but does NOT unlink the
/// shared memory files from /dev/shm. Use unlinkSharedMemory() for that.
///
/// @post All memory unmapped and file descriptors closed
/// @post initialized flag set to false
////////////////////////////////////////////////////////////////////////////////
void EKIWriter::cleanup() {
    if (data_map) {
        munmap(data_map, data_size);
        data_map = nullptr;
    }
    if (config_map) {
        munmap(config_map, sizeof(EKIConfigFull));
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

////////////////////////////////////////////////////////////////////////////////
/// @brief Unlink shared memory segments from filesystem
///
/// @details
/// Removes shared memory files from /dev/shm. Should be called at program
/// exit after all readers (Python) have finished.
///
/// Note: Does NOT unlink ensemble observation files because Python needs
/// them after C++ exits.
///
/// @post /dev/shm/ldm_eki_config removed
/// @post /dev/shm/ldm_eki_data removed
////////////////////////////////////////////////////////////////////////////////
void EKIWriter::unlinkSharedMemory() {
    shm_unlink(SHM_CONFIG_NAME);
    shm_unlink(SHM_DATA_NAME);
    // Don't unlink ensemble observation files - Python needs them!
    // shm_unlink(SHM_ENSEMBLE_OBS_CONFIG_NAME);
    // shm_unlink(SHM_ENSEMBLE_OBS_DATA_NAME);
    std::cout << Color::CYAN << "[IPC] " << Color::RESET
              << "Shared memory unlinked (ensemble obs kept for Python)\n";
}

} // namespace LDM_EKI_IPC

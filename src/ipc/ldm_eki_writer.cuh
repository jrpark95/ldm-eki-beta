////////////////////////////////////////////////////////////////////////////////
/// @file    ldm_eki_writer.cuh
/// @brief   IPC writer for transmitting observation data from C++ to Python
/// @details Manages POSIX shared memory segments to send receptor observations
///          from the LDM forward model to the Python EKI inversion process.
///          Supports both initial "truth" observations and iterative ensemble
///          observations during the Kalman filter optimization loop.
///
/// @author  Juryong Park
/// @date    2025
////////////////////////////////////////////////////////////////////////////////

#pragma once
#ifndef LDM_EKI_WRITER_CUH
#define LDM_EKI_WRITER_CUH

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include "colors.h"

// Forward declarations
class MemoryDoctor;
extern MemoryDoctor g_memory_doctor;

// Forward declaration of EKIConfig (defined in ldm_struct.cuh)
struct EKIConfig;

namespace LDM_EKI_IPC {

////////////////////////////////////////////////////////////////////////////////
/// @name Shared Memory Segment Names
/// @{
/// @note All segments reside in /dev/shm (tmpfs)
////////////////////////////////////////////////////////////////////////////////

constexpr const char* SHM_CONFIG_NAME = "/ldm_eki_config";
constexpr const char* SHM_DATA_NAME = "/ldm_eki_data";
constexpr const char* SHM_ENSEMBLE_OBS_CONFIG_NAME = "/ldm_eki_ensemble_obs_config";
constexpr const char* SHM_ENSEMBLE_OBS_DATA_NAME = "/ldm_eki_ensemble_obs_data";
constexpr const char* SHM_TRUE_EMISSIONS_NAME = "/ldm_eki_true_emissions";

/// @}

////////////////////////////////////////////////////////////////////////////////
/// @name Configuration Structures
/// @{
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// @struct EKIConfigBasic
/// @brief  Minimal EKI configuration (12 bytes)
/// @details Backward-compatible basic configuration for legacy Python code
////////////////////////////////////////////////////////////////////////////////
struct EKIConfigBasic {
    int32_t ensemble_size;  ///< Number of ensemble members (e.g., 100)
    int32_t num_receptors;  ///< Number of spatial receptors (e.g., 3)
    int32_t num_timesteps;  ///< Number of temporal observation points (e.g., 24)
};

////////////////////////////////////////////////////////////////////////////////
/// @struct EKIConfigFull
/// @brief  Complete EKI configuration (84 bytes)
/// @details Full configuration including all algorithm parameters, options,
///          and metadata needed by the Python EKI optimization process.
///
/// Memory Layout (v1.0 Release - Simplified):
/// - Bytes 0-11  : Basic dimensions (12 bytes)
/// - Bytes 12-35 : Algorithm parameters (24 bytes)
/// - Bytes 36-75 : Option strings (40 bytes = 5 strings × 8 bytes)
/// - Bytes 76-83 : Memory Doctor mode (8 bytes)
///
/// Separate shared memory segments:
/// - /dev/shm/ldm_eki_true_emissions: True emission time series [num_timesteps × float]
///
/// Hardcoded values (not in IPC structure):
/// - GPU acceleration: Always enabled (CUDA for forward, CuPy for inverse)
/// - Source location: Always "Fixed" (known position)
/// - Number of sources: Always 1 (single source only)
///
/// @note Total size is 84 bytes (was 80 bytes before decay_constant added)
/// @note All strings are null-terminated with max length 7 chars + null
/// @note Deprecated fields removed: time_days, inverse_time_interval,
///       receptor_error, receptor_mda, num_source, num_gpu, gpu_forward,
///       gpu_inverse, source_location
////////////////////////////////////////////////////////////////////////////////
struct EKIConfigFull {
    // Basic dimensions (12 bytes)
    int32_t ensemble_size;
    int32_t num_receptors;
    int32_t num_timesteps;

    // Algorithm parameters (24 bytes)
    int32_t iteration;               ///< Maximum EKI iterations (e.g., 10)
    float renkf_lambda;              ///< REnKF regularization parameter
    float noise_level;               ///< Observation noise level
    float time_interval;             ///< EKI time interval (e.g., 15.0 minutes)
    float prior_constant;            ///< Prior emission constant (e.g., 1.5e+8 Bq)
    float decay_constant;            ///< Nuclide decay constant λ [s⁻¹] (e.g., Kr-88: 6.78e-5)

    // Option strings (40 bytes = 5 strings × 8 bytes)
    char perturb_option[8];    ///< Perturbation option: "On"/"Off"
    char adaptive_eki[8];      ///< Adaptive EKI: "On"/"Off"
    char localized_eki[8];     ///< Localized EKI: "On"/"Off"
    char regularization[8];    ///< Regularization: "On"/"Off"
    char time_unit[8];         ///< Time unit: "minutes"/"hours"

    // Memory Doctor Mode (8 bytes)
    char memory_doctor[8];     ///< Debug mode: "On"/"Off"

    // Total: 12 + 24 + 40 + 8 = 84 bytes (no padding needed)

    // Note: true_emissions array sent via separate shared memory segment
    // (/dev/shm/ldm_eki_true_emissions) to avoid fixed-size allocation
};

////////////////////////////////////////////////////////////////////////////////
/// @struct EKIDataHeader
/// @brief  Header for observation data arrays (12 bytes + variable data)
/// @details Prepends observation data to provide metadata and ready status
///
/// Memory Layout:
/// - [0-11 bytes]      : Header (status, dimensions)
/// - [12+ bytes]       : Float data (rows × cols elements)
////////////////////////////////////////////////////////////////////////////////
struct EKIDataHeader {
    int32_t status;      ///< 0=writing (incomplete), 1=ready (complete)
    int32_t rows;        ///< Number of receptors
    int32_t cols;        ///< Number of timesteps
    // float data[] follows immediately after header
};

/// @}

////////////////////////////////////////////////////////////////////////////////
/// @class EKIWriter
/// @brief IPC writer for transmitting observation data from LDM to Python
///
/// @details
/// This class manages POSIX shared memory segments to transfer receptor
/// observation data from the C++/CUDA forward model to the Python EKI
/// inversion process. It handles three types of data transfers:
///
/// 1. **Configuration**: Full EKI algorithm parameters and settings
/// 2. **Initial Observations**: "Truth" simulation observations
/// 3. **Ensemble Observations**: Per-iteration ensemble member observations
///
/// Communication Protocol:
/// ```
/// 1. C++ calls initialize() → creates config + data segments
/// 2. C++ writes initial observations → Python reads them
/// 3. Python creates ensemble priors → writes to separate segments
/// 4. [LOOP] C++ writes ensemble observations → Python updates states
/// ```
///
/// Shared Memory Segments:
/// - `/dev/shm/ldm_eki_config` : Full configuration (84 bytes)
/// - `/dev/shm/ldm_eki_data` : Initial observations (header + data)
/// - `/dev/shm/ldm_eki_ensemble_obs_config` : Ensemble dimensions (12 bytes)
/// - `/dev/shm/ldm_eki_ensemble_obs_data` : Ensemble observations (variable size)
///
/// Data Format (Row-Major):
/// - Initial observations: [receptors × timesteps] matrix
/// - Ensemble observations: [ensembles × receptors × timesteps] tensor
///
/// @note Thread-safe for single writer (this class), multiple readers (Python)
/// @note All data uses row-major (C-style) layout for compatibility
///
/// @see EKIReader for reciprocal Python→C++ data transfer
/// @see Memory Doctor system for IPC debugging and validation
///
/// @par Example Usage:
/// @code
/// EKIWriter writer;
/// writer.initialize(eki_config, num_timesteps);
/// writer.writeObservations(obs_data, num_receptors, num_timesteps);
/// writer.initializeEnsembleObservations(ensemble_size, num_receptors, num_timesteps);
/// // ... iteration loop ...
/// writer.writeEnsembleObservations(ens_obs, ensemble_size, num_receptors, num_timesteps, iter);
/// writer.cleanup();
/// @endcode
////////////////////////////////////////////////////////////////////////////////
class EKIWriter {
private:
    int config_fd;         ///< File descriptor for configuration segment
    int data_fd;           ///< File descriptor for initial observation data segment
    void* config_map;      ///< Memory-mapped configuration structure (EKIConfigFull*)
    void* data_map;        ///< Memory-mapped data buffer (EKIDataHeader + float[])
    size_t data_size;      ///< Size of data segment [bytes]
    bool initialized;      ///< Initialization state flag

public:
    ////////////////////////////////////////////////////////////////////////////////
    /// @name Constructor / Destructor
    /// @{
    ////////////////////////////////////////////////////////////////////////////////

    EKIWriter();
    ~EKIWriter();

    /// @}

    ////////////////////////////////////////////////////////////////////////////////
    /// @name Initialization
    /// @{
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    /// @brief Initialize shared memory segments with full configuration
    ///
    /// @details
    /// Creates and maps POSIX shared memory for both configuration and initial
    /// observation data. This is the first step in the IPC protocol and must be
    /// called before any write operations.
    ///
    /// Steps:
    /// 1. Calculate data segment size from config dimensions
    /// 2. Create config segment (/ldm_eki_config) and map EKIConfigFull
    /// 3. Write all EKI parameters to config segment
    /// 4. Create data segment (/ldm_eki_data) with header + observation space
    /// 5. Initialize data header with dimensions and status=0 (writing)
    ///
    /// @param[in] eki_config     Complete EKI configuration structure
    /// @param[in] num_timesteps  Number of simulation timesteps
    ///
    /// @return true if initialization successful, false on any error
    ///
    /// @pre POSIX shared memory support available (/dev/shm mounted)
    /// @post Shared memory segments created and mapped
    /// @post Configuration data written to /dev/shm/ldm_eki_config
    /// @post Data segment allocated at /dev/shm/ldm_eki_data
    ///
    /// @note Automatically calculates data segment size as:
    ///       sizeof(EKIDataHeader) + (num_receptors × num_timesteps × sizeof(float))
    /// @warning Must be called before any write operations
    /// @warning Calling multiple times without cleanup() will fail
    ///
    /// @par Output:
    /// Prints colored summary of configuration to console
    ////////////////////////////////////////////////////////////////////////////////
    bool initialize(const ::EKIConfig& eki_config, int num_timesteps);

    /// @}

    ////////////////////////////////////////////////////////////////////////////////
    /// @name Initial Observations
    /// @{
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    /// @brief Write initial observation matrix to shared memory
    ///
    /// @details
    /// Transfers the "truth" simulation observation data from C++ to Python.
    /// This is typically called once after the initial forward simulation to
    /// provide reference observations for the Kalman filter.
    ///
    /// Data Layout (Row-Major):
    /// ```
    /// [receptor_0_time_0, receptor_0_time_1, ..., receptor_0_time_T,
    ///  receptor_1_time_0, receptor_1_time_1, ..., receptor_1_time_T,
    ///  ...,
    ///  receptor_R_time_0, receptor_R_time_1, ..., receptor_R_time_T]
    /// ```
    ///
    /// Protocol:
    /// 1. Set header status = 0 (writing)
    /// 2. Copy observations to data segment
    /// 3. Log to Memory Doctor if enabled
    /// 4. Set header status = 1 (ready) → signals Python
    ///
    /// @param[in] observations  Observation matrix [receptors × timesteps]
    /// @param[in] rows          Number of receptors
    /// @param[in] cols          Number of timesteps
    ///
    /// @return true if write successful, false on error
    ///
    /// @pre initialize() must have been called successfully
    /// @pre Dimensions must match those provided to initialize()
    /// @post Data written to /dev/shm/ldm_eki_data
    /// @post Header status set to 1 (ready for Python to read)
    ///
    /// @note Uses memcpy for efficient bulk transfer
    /// @note Validates dimensions against header before writing
    ///
    /// @par Output:
    /// Prints data size and dimensions to console
    ////////////////////////////////////////////////////////////////////////////////
    bool writeObservations(const float* observations, int rows, int cols);

    /// @}

    ////////////////////////////////////////////////////////////////////////////////
    /// @name Ensemble Observations
    /// @{
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    /// @brief Initialize ensemble observation shared memory segments
    ///
    /// @details
    /// Creates separate shared memory segments for ensemble observation
    /// configuration and data. Must be called once before the first call to
    /// writeEnsembleObservations().
    ///
    /// Creates two segments:
    /// - `/ldm_eki_ensemble_obs_config`: EKIConfigBasic (12 bytes)
    /// - `/ldm_eki_ensemble_obs_data`: Ready for variable-size data
    ///
    /// @param[in] ensemble_size   Number of ensemble members (e.g., 100)
    /// @param[in] num_receptors   Number of spatial receptors (e.g., 3)
    /// @param[in] num_timesteps   Number of temporal observations (e.g., 24)
    ///
    /// @return true if successful, false on any error
    ///
    /// @pre initialize() must have been called
    /// @post /ldm_eki_ensemble_obs_config created and written
    /// @post Ready for subsequent writeEnsembleObservations() calls
    ///
    /// @note Config segment is closed immediately after writing
    /// @note Data segment is created/truncated on each writeEnsembleObservations()
    ///
    /// @par Output:
    /// Prints ensemble dimensions to console
    ////////////////////////////////////////////////////////////////////////////////
    bool initializeEnsembleObservations(int ensemble_size, int num_receptors, int num_timesteps);

    ////////////////////////////////////////////////////////////////////////////////
    /// @brief Write ensemble observation tensor to shared memory
    ///
    /// @details
    /// Transfers ensemble member observations from C++ to Python during each
    /// EKI iteration. Each ensemble member has its own observation matrix.
    ///
    /// Data Layout (Row-Major):
    /// ```
    /// [ens_0_recep_0_time_0, ..., ens_0_recep_0_time_T,
    ///  ens_0_recep_1_time_0, ..., ens_0_recep_1_time_T,
    ///  ...,
    ///  ens_0_recep_R_time_0, ..., ens_0_recep_R_time_T,
    ///  ens_1_recep_0_time_0, ..., (repeat for all ensembles)]
    /// ```
    ///
    /// Total size: ensemble_size × num_receptors × num_timesteps × sizeof(float)
    ///
    /// Steps:
    /// 1. Calculate data size
    /// 2. Create/truncate /ldm_eki_ensemble_obs_data segment
    /// 3. Map segment to memory
    /// 4. Copy observations with memcpy
    /// 5. Calculate statistics (min/max/mean) for validation
    /// 6. Log to Memory Doctor with iteration tracking
    /// 7. Unmap and close (Python will re-open)
    ///
    /// @param[in] observations    3D observation tensor (row-major, flattened)
    /// @param[in] ensemble_size   Number of ensemble members
    /// @param[in] num_receptors   Number of receptors
    /// @param[in] num_timesteps   Number of timesteps
    /// @param[in] iteration       Current EKI iteration number (for logging)
    ///
    /// @return true if write successful, false on any error
    ///
    /// @pre initializeEnsembleObservations() must have been called
    /// @post Data written to /dev/shm/ldm_eki_ensemble_obs_data
    /// @post Statistics logged to console
    ///
    /// @note Creates new segment each iteration (O_TRUNC flag)
    /// @note Python expected to read data before next write
    ///
    /// @par Output:
    /// Prints data size, shape, and statistics (range, mean) to console
    ///
    /// @performance Typical transfer: 10-100 KB in < 1ms
    ////////////////////////////////////////////////////////////////////////////////
    bool writeEnsembleObservations(const float* observations, int ensemble_size,
                                   int num_receptors, int num_timesteps, int iteration = -1);

    /// @}

    ////////////////////////////////////////////////////////////////////////////////
    /// @name True Emissions Transfer
    /// @{
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    /// @brief Write true emission time series to shared memory
    ///
    /// @details
    /// Transfers the true emission time series from C++ to Python via a
    /// separate shared memory segment. This array is read from eki.conf
    /// and must be passed to Python for the EKI optimization.
    ///
    /// Data Layout:
    /// - [emission_time_0, emission_time_1, ..., emission_time_T]
    ///
    /// Segment:
    /// - `/dev/shm/ldm_eki_true_emissions`
    ///
    /// Size: num_timesteps × sizeof(float)
    ///
    /// @param[in] emissions      True emission time series array
    /// @param[in] num_timesteps  Number of timesteps (array length)
    ///
    /// @return true if write successful, false on any error
    ///
    /// @pre initialize() must have been called
    /// @post Data written to /dev/shm/ldm_eki_true_emissions
    /// @post Statistics logged to console
    ///
    /// @note Creates new segment each call (O_TRUNC flag)
    /// @note Python expected to read data after this call
    ///
    /// @par Output:
    /// Prints data size and statistics (range, sum) to console
    ///
    /// @performance Typical transfer: < 1 KB in < 1ms
    ////////////////////////////////////////////////////////////////////////////////
    bool writeTrueEmissions(const std::vector<float>& emissions);

    /// @}

    ////////////////////////////////////////////////////////////////////////////////
    /// @name Configuration Retrieval
    /// @{
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    /// @brief Get current basic configuration dimensions
    ///
    /// @details
    /// Retrieves the basic configuration dimensions that were set during
    /// initialization. Useful for validation and error checking.
    ///
    /// @param[out] ensemble_size   Number of ensemble members
    /// @param[out] num_receptors   Number of receptors
    /// @param[out] num_timesteps   Number of timesteps
    ///
    /// @return true if config available, false if not initialized
    ////////////////////////////////////////////////////////////////////////////////
    bool getConfig(int& ensemble_size, int& num_receptors, int& num_timesteps);

    /// @}

    ////////////////////////////////////////////////////////////////////////////////
    /// @name Cleanup
    /// @{
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    /// @brief Cleanup resources and unmap memory
    ///
    /// @details
    /// Unmaps all memory-mapped segments and closes file descriptors.
    /// Does NOT unlink shared memory files from /dev/shm - use
    /// unlinkSharedMemory() for that.
    ///
    /// Steps:
    /// 1. Unmap data segment if mapped
    /// 2. Unmap config segment if mapped
    /// 3. Close data file descriptor
    /// 4. Close config file descriptor
    /// 5. Set initialized flag to false
    ///
    /// @post All file descriptors closed
    /// @post All memory mappings released
    /// @post initialized flag set to false
    /// @post Shared memory files remain in /dev/shm
    ///
    /// @note Safe to call multiple times (idempotent)
    /// @note Automatically called by destructor
    /// @note Does NOT delete shared memory files from filesystem
    ////////////////////////////////////////////////////////////////////////////////
    void cleanup();

    ////////////////////////////////////////////////////////////////////////////////
    /// @brief Unlink shared memory segments from filesystem
    ///
    /// @details
    /// Removes shared memory files from /dev/shm. Should be called at program
    /// exit after all readers (Python) have finished accessing the data.
    ///
    /// Unlinks:
    /// - /dev/shm/ldm_eki_config
    /// - /dev/shm/ldm_eki_data
    ///
    /// Does NOT unlink:
    /// - /dev/shm/ldm_eki_ensemble_obs_config (Python needs it)
    /// - /dev/shm/ldm_eki_ensemble_obs_data (Python needs it)
    ///
    /// @note Static method - can be called without instance
    /// @warning Call only when certain no processes need the data
    /// @warning Premature unlinking will cause Python read failures
    ///
    /// @par Output:
    /// Prints confirmation message to console
    ////////////////////////////////////////////////////////////////////////////////
    static void unlinkSharedMemory();

    /// @}
};

} // namespace LDM_EKI_IPC

#endif // LDM_EKI_WRITER_CUH

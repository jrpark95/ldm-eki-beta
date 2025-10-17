/******************************************************************************
 * @file ldm.cuh
 * @brief Main header file for Lagrangian Dispersion Model (LDM) class
 *
 * This header defines the LDM class, which is the central orchestrator for
 * GPU-accelerated atmospheric dispersion simulations. The class manages:
 * - Particle tracking with CUDA kernels
 * - Meteorological data loading and interpolation
 * - Radioactive decay chains (CRAM method)
 * - EKI ensemble inversion coupling
 * - VTK visualization output
 *
 * Architecture:
 * - Host (CPU): Configuration, I/O, orchestration
 * - Device (GPU): Particle advection, turbulent diffusion, decay kernels
 * - Hybrid: Meteorological data cached on both host and device
 *
 * Key Design Patterns:
 * - Singleton: NuclideConfig, ConfigReader
 * - Data-oriented design: Large particle arrays processed in parallel
 * - Cache-friendly: Preload meteorology for EKI iterations
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/

#pragma once

// ===========================================================================
// System Headers
// ===========================================================================

// CUDA runtime and device libraries
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Standard C++ libraries
#include <iostream>
#include <cstdio>
#include <vector>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <math.h>
#include <limits>
#include <float.h>

// Timing and concurrency
#include <chrono>
#include <random>
#include <tuple>
#include <future>
#include <thread>
#include <mutex>
#include <atomic>


#include "../data/config/ldm_struct.cuh"
#include "../data/config/ldm_config.cuh"
//#include "ldm_cram.cuh"
#include "../physics/ldm_cram2.cuh"  // Must be included early for N_NUCLIDES macro (used at line 235)

#include <math_constants.h>
#include <vector_types.h>

#ifdef _WIN32
    #include <direct.h>
#else
    #include <sys/types.h>
    #include <sys/stat.h>
#endif

// ===========================================================================
// Global Physics Model Switches
// ===========================================================================

// Physics model flags (loaded from physics.conf)
// These control which physical processes are simulated:
extern int g_turb_switch;  // Turbulent diffusion (0=off, 1=on)
extern int g_drydep;       // Dry deposition (0=off, 1=on)
extern int g_wetdep;       // Wet deposition (0=off, 1=on)
extern int g_raddecay;     // Radioactive decay (0=off, 1=on)

// ===========================================================================
// Device Memory Organization Notes
// ===========================================================================

// CUDA Constant Memory Removal:
// - Scalar constants previously in __constant__ memory are now passed
//   via KernelScalars struct to avoid "invalid device symbol" errors
//   in non-RDC (relocatable device code) compilation mode
// - Arrays (d_flex_hgt, T_matrix) allocated with cudaMalloc for flexibility
// - This approach improves portability across CUDA architectures

// ===========================================================================
// Configuration Structures (Modern Architecture)
// ===========================================================================

/******************************************************************************
 * @struct SimulationConfig
 * @brief Core simulation parameters (time integration, particles, domain)
 *
 * Loaded from: simulation.conf
 *
 * Time Integration:
 * - timeEnd: Total simulation duration [seconds]
 * - deltaTime: Timestep for particle advection [seconds]
 * - outputFrequency: VTK output interval [timesteps]
 *
 * Particle Configuration:
 * - numParticles: Total number of particles released
 * - settlingVelocity: Gravitational settling speed [m/s]
 * - cunninghamFactor: Slip correction factor (dimensionless)
 *
 * Meteorology Source:
 * - isGFS: true = GFS global reanalysis, false = LDAPS regional model
 * - gfsIndex: Time index in GFS file sequence
 *
 * Dispersion Model:
 * - isRural: true = rural terrain, false = urban terrain
 * - isPasquillGifford: true = P-G scheme, false = similarity theory
 *
 * @note Parameters validated during loadSimulationConfig()
 *****************************************************************************/
struct SimulationConfig {
    float timeEnd;                      // Simulation end time [s]
    float deltaTime;                    // Integration timestep [s]
    int outputFrequency;                // VTK output interval [timesteps]
    int numParticles;                   // Total particle count
    int gfsIndex = 0;                   // GFS file time index
    bool isRural;                       // Rural vs urban terrain
    bool isPasquillGifford;             // P-G vs similarity theory
    bool isGFS;                         // GFS vs LDAPS meteorology
    float settlingVelocity = 0.0;       // Gravitational settling [m/s]
    float cunninghamFactor = 0.0;       // Slip correction factor
    bool fixedScrollOutput = true;      // Terminal fixed-position output
};

struct MPIConfig {
    int rank = 1;
    int size = 1;
    std::string species[4];
    float decayConstants[4];
    float depositionVelocities[4];
    float particleSizes[4];
    float particleDensities[4];
    float sizeStandardDeviations[4];
};

struct EKIConfig {
    bool mode = false;
    float time_interval = 15.0f;
    std::string time_unit = "minutes";
    int num_receptors = 0;
    std::vector<std::pair<float, float>> receptor_locations;
    float receptor_capture_radius = 0.01f;  // degrees
    std::vector<float> true_emissions;
    std::vector<float> prior_emissions;
    std::string prior_mode = "constant";
    float prior_constant = 1.5e+8f;
    int ensemble_size = 50;
    float noise_level = 0.01f;

    // Python EKI algorithm parameters
    int iteration = 10;
    std::string perturb_option = "Off";
    std::string adaptive_eki = "Off";
    std::string localized_eki = "Off";
    std::string regularization = "On";
    float renkf_lambda = 1.0f;

    // ========== HARDCODED SYSTEM SETTINGS (v1.0 Release) ==========
    // These values are fixed for production use and not configurable
    // GPU acceleration: Always enabled (CuPy for Python, CUDA for C++)
    // Source location: Always "Fixed" (known position, estimate emissions only)
    // Number of sources: Always 1 (multi-source inversion not supported)

    // Memory Doctor Mode for IPC debugging
    bool memory_doctor_mode = false;
};

/******************************************************************************
 * @struct EKIMeteorologicalData
 * @brief Preloaded meteorological data cache for EKI ensemble iterations
 *
 * Performance Optimization:
 * EKI requires running 10-100 iterations, each simulating 50-100 ensemble
 * members. Loading meteorological files from disk for every iteration would
 * create severe I/O bottleneck. This structure preloads ALL timesteps into
 * memory once before iteration loop begins.
 *
 * Memory Layout:
 * - Host: std::vector<FlexPres*> - All timesteps in CPU RAM
 * - Device: FlexPres** - Array of GPU pointers to each timestep
 * - Interpolation slots: Past/future timestep buffers for temporal interpolation
 *
 * Typical Memory Usage (100 ensembles, 48 timesteps, GFS 26 levels):
 * - FlexPres: ~200 MB per timestep × 48 = 9.6 GB
 * - FlexUnis: ~50 MB per timestep × 48 = 2.4 GB
 * - Total: ~12 GB GPU memory (requires high-end GPU)
 *
 * Lifecycle:
 * 1. Preload: Call LDM::preloadAllEKIMeteorologicalData() before iteration loop
 * 2. Access: Kernels access via device_flex_pres_data[timestep_index]
 * 3. Cleanup: Automatic via destructor or explicit cleanup()
 *
 * @note Requires GPU with sufficient memory (12+ GB recommended)
 * @note Initialization sets is_initialized = true for safety checks
 *
 * @see LDM::preloadAllEKIMeteorologicalData()
 * @see LDM::cleanupEKIMeteorologicalData()
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
struct EKIMeteorologicalData {
    int num_time_steps = 0;  // Number of preloaded timesteps

    // Host memory: All timesteps cached in CPU RAM
    std::vector<FlexPres*> host_flex_pres_data;       // Pressure-level fields
    std::vector<FlexUnis*> host_flex_unis_data;       // Surface fields
    std::vector<std::vector<float>> host_flex_hgt_data; // Height levels

    // Device memory: Array of GPU pointers (one per timestep)
    FlexPres** device_flex_pres_data = nullptr;  // GPU pointer array
    FlexUnis** device_flex_unis_data = nullptr;  // GPU pointer array
    float** device_flex_hgt_data = nullptr;      // GPU pointer array

    // Temporal interpolation buffers (past and future timesteps)
    FlexPres* ldm_pres0_slot = nullptr;  // Past timestep buffer
    FlexUnis* ldm_unis0_slot = nullptr;  // Past timestep buffer
    FlexPres* ldm_pres1_slot = nullptr;  // Future timestep buffer
    FlexUnis* ldm_unis1_slot = nullptr;  // Future timestep buffer

    // Memory allocation sizes (bytes)
    size_t pres_data_size = 0;  // FlexPres structure size
    size_t unis_data_size = 0;  // FlexUnis structure size
    size_t hgt_data_size = 0;   // Height array size

    // Initialization state flag (safety check)
    bool is_initialized = false;

    // Destructor: Automatic cleanup when object goes out of scope
    ~EKIMeteorologicalData();

    // Explicit cleanup function (can be called before destructor)
    void cleanup();
};

// Global variables - always declared as extern in header
extern SimulationConfig g_sim;
extern MPIConfig g_mpi;
extern EKIConfig g_eki;
extern ConfigReader g_config;
extern EKIMeteorologicalData g_eki_meteo;
extern std::vector<float> flex_hgt;

// Log file handle (declared in main_eki.cu)
// Other modules can write directly to this for log-only output
extern std::ofstream* g_log_file;

// 물리/격자 상수 정리 (계산 로직 변경 없음)
namespace Constants {
    // 격자 상수
    constexpr float LDAPS_E = 132.36f, LDAPS_W = 121.06f, LDAPS_N = 43.13f, LDAPS_S = 32.20f;
    constexpr int dimX = 602, dimY = 781, dimZ_pres = 24, dimZ_etas = 71;
    constexpr int dimX_GFS = 720, dimY_GFS = 361, dimZ_GFS = 26;
    constexpr int time_interval = 10800;
    
    // 물리 상수
    constexpr float d_trop = 50.0f, d_strat = 0.1f, turbmesoscale = 0.16f, r_earth = 6371000.0f;
    constexpr float _myl = 1.81e-5f, _nyl = 0.15e-4f, _lam = 6.53e-8f, _kb = 1.38e-23f;
    constexpr float _eps = 1.2e-38f, _Tr = 293.15f, _rair = 287.05f, _ga = 9.81f, _href = 15.0f;
    constexpr int _nspec = 19;
    constexpr int NI = 11;
    // Note: N_NUCLIDES is defined as a macro in ldm_cram2.cuh (included below)
}

using namespace Constants;

// Global variables (like LDM-CRAM4)
// float time_end;
// float dt;
// int freq_output;
// int nop;
// int gfs_idx = 0;
// bool isRural;
// bool isPG;
// bool isGFS;
// float vsetaver = 0.0;
// float cunningham = 0.0;
#define time_end (g_sim.timeEnd)
#define dt (g_sim.deltaTime)
#define freq_output (g_sim.outputFrequency)
#define nop (g_sim.numParticles)
#define gfs_idx (g_sim.gfsIndex)
#define isRural (g_sim.isRural)
#define isPG (g_sim.isPasquillGifford)
#define isGFS (g_sim.isGFS)
#define vsetaver (g_sim.settlingVelocity)
#define cunningham (g_sim.cunninghamFactor)


// Device constant scalars removed - now passed via KernelScalars struct parameter
// This avoids "invalid device symbol" errors in non-RDC compilation mode

// Device arrays - REMOVED: d_flex_hgt now allocated with cudaMalloc in LDM class

__device__ double d_uWind[512];
__device__ double d_vWind[512];

class LDM
{
private:

    PresData* device_meteorological_data_pres;  // Used in constructor/destructor
    UnisData* device_meteorological_data_unis;  // Used in constructor/destructor
    EtasData* device_meteorological_data_etas;  // Used in constructor/destructor

    FlexUnis* device_meteorological_flex_unis0;
    FlexPres* device_meteorological_flex_pres0;
    FlexUnis* device_meteorological_flex_unis1;
    FlexPres* device_meteorological_flex_pres1;
    FlexUnis* device_meteorological_flex_unis2;
    FlexPres* device_meteorological_flex_pres2;

    // CRAM decay matrix (GPU memory)
    float* d_T_matrix;  // Decay transition matrix [N_NUCLIDES × N_NUCLIDES]

    // Height data (GPU memory)
    float* d_flex_hgt;  // Vertical height levels [dimZ_GFS = 50 elements]

    // EKI observation system variables
    float* d_receptor_lats;         // GPU memory for receptor latitudes
    float* d_receptor_lons;         // GPU memory for receptor longitudes
    float* d_receptor_dose;         // GPU memory for receptor dose accumulation
    std::vector<std::vector<float>> eki_observations; // [time_step][receptor] dose measurements
    std::vector<float> eki_observation_times;         // Actual observation times (seconds)
    int eki_observation_count;      // Number of observation time steps taken

    // EKI ensemble observation system variables
    float* d_ensemble_dose;         // GPU memory for ensemble dose [num_ensembles × num_receptors × num_timesteps]
    int* d_ensemble_particle_count; // GPU memory for particle counts [num_ensembles × num_receptors × num_timesteps]
    std::vector<std::vector<std::vector<float>>> eki_ensemble_observations; // [ensemble][time_step][receptor]
    std::vector<std::vector<std::vector<int>>> eki_ensemble_particle_counts; // [ensemble][time_step][receptor] - particle counts

    // Single mode particle count
    int* d_receptor_particle_count; // GPU memory for particle counts [num_receptors]
    std::vector<std::vector<int>> eki_particle_counts; // [time_step][receptor] - particle counts

    // float4* host_unisA0; // HMIX, USTR, WSTR, OBKL
    // float4* host_unisB0; // VDEP, LPREC, CPREC, TCC
    // float4* host_unisA1;
    // float4* host_unisB1;
    // float4* host_unisA2;
    // float4* host_unisB2;

    // cudaArray* d_unisArrayA0; //
    // cudaArray* d_unisArrayB0; //
    // cudaArray* d_unisArrayA1; //
    // cudaArray* d_unisArrayB1; //
    // cudaArray* d_unisArrayA2; //
    // cudaArray* d_unisArrayB2; //

    // cudaChannelFormatDesc channelDesc2D = cudaCreateChannelDesc<float4>();
    // cudaChannelFormatDesc channelDesc3D = cudaCreateChannelDesc<float4>();

    // float4* host_presA0; // DRHO, RHO, TT, QV
    // float4* host_presB0; // UU, VV, WW, 0.0 
    // float4* host_presA1;
    // float4* host_presB1;
    // float4* host_presA2;
    // float4* host_presB2;

    // cudaArray* d_presArrayA0; //
    // cudaArray* d_presArrayB0; //
    // cudaArray* d_presArrayA1; //
    // cudaArray* d_presArrayB1; //
    // cudaArray* d_presArrayA2; //
    // cudaArray* d_presArrayB2; //

    std::vector<Source> sources;  // Used in initialization
    std::vector<float> decayConstants;  // Used in initialization
    std::vector<float> drydepositionVelocity;  // Used in initialization
    std::vector<Concentration> concentrations;  // Used in initialization

    cudaTextureObject_t m_texUnisA0 = 0;
    cudaTextureObject_t m_texUnisA1 = 0;
    cudaTextureObject_t m_texUnisA2 = 0;

    cudaTextureObject_t m_texUnisB0 = 0;
    cudaTextureObject_t m_texUnisB1 = 0;
    cudaTextureObject_t m_texUnisB2 = 0;

    cudaTextureObject_t m_texPresA0 = 0;
    cudaTextureObject_t m_texPresA1 = 0;
    cudaTextureObject_t m_texPresA2 = 0;

    cudaTextureObject_t m_texPresB0 = 0;
    cudaTextureObject_t m_texPresB1 = 0;
    cudaTextureObject_t m_texPresB2 = 0;
    
public:

    LDM();
    ~LDM();

    float minX, minY, maxX, maxY;

    // Make EKI config public for main_eki.cu access
    EKIConfig& getEKIConfig() { return g_eki; }
    std::vector<std::vector<std::vector<float>>>& getEKIEnsembleObservations() { return eki_ensemble_observations; }
    float *d_minX, *d_minY, *d_maxX, *d_maxY;

    // Grid receptor debug mode variables (public for main_receptor_debug.cu access)
    bool is_grid_receptor_mode = false;  // Flag to enable grid receptor mode
    int grid_count = 0;                  // Number of receptors in each direction from source
    float grid_spacing = 0.0f;           // Distance between receptors in degrees
    int grid_receptor_total = 0;         // Total number of grid receptors
    float* d_grid_receptor_lats = nullptr;         // GPU memory for grid receptor latitudes
    float* d_grid_receptor_lons = nullptr;         // GPU memory for grid receptor longitudes
    float* d_grid_receptor_dose = nullptr;         // GPU memory for grid receptor dose accumulation
    int* d_grid_receptor_particle_count = nullptr; // GPU memory for grid particle counts
    std::vector<std::vector<float>> grid_receptor_observations;  // [receptor][time_step] dose
    std::vector<std::vector<int>> grid_receptor_particle_counts; // [receptor][time_step] particles
    std::vector<float> grid_observation_times;                   // Observation times (seconds)

    // std::vector<float> U_flat;  // Unused - only used in lkd/val functions
    // std::vector<float> V_flat;  // Unused - only used in lkd/val functions
    // std::vector<float> W_flat;  // Unused - only used in lkd/val functions

    // float* d_U_flat = nullptr;  // Unused - only used in lkd/val functions
    // float* d_V_flat = nullptr;  // Unused - only used in lkd/val functions
    // float* d_W_flat = nullptr;  // Unused - only used in lkd/val functions

    // std::vector<float> U_flat_next;  // Unused - only used in lkd/val functions
    // std::vector<float> V_flat_next;  // Unused - only used in lkd/val functions
    // std::vector<float> W_flat_next;  // Unused - only used in lkd/val functions

    // float* d_U_flat_next = nullptr;  // Unused - only used in lkd/val functions
    // float* d_V_flat_next = nullptr;  // Unused - only used in lkd/val functions
    // float* d_W_flat_next = nullptr;  // Unused - only used in lkd/val functions

    std::chrono::high_resolution_clock::time_point timerStart, timerEnd;
    std::chrono::high_resolution_clock::time_point stepStart, stepEnd;

    // Ensemble mode control variables
    bool is_ensemble_mode;          // Flag to indicate ensemble simulation mode
    int ensemble_size;              // Number of ensemble members
    int ensemble_num_states;        // Number of state timesteps per ensemble
    std::vector<int> selected_ensemble_ids; // Random ensemble IDs for VTK output (3 ensembles)
    bool enable_vtk_output;         // Runtime flag to enable VTK output (controlled dynamically)

    // VTK output configuration (from simulation.conf)
    bool config_enable_single_mode_vtk;   // User config: enable VTK for single mode
    bool config_enable_ensemble_mode_vtk; // User config: enable VTK for ensemble mode

    __device__ __host__ struct LDMpart{

        float x, y, z;                       // Essential: Position coordinates
        float u, v, w;                       // Essential: Velocity components
        float up, vp, wp;                    // Essential: Turbulent velocity components (used in kernels)
        // float um, vm, wm;                // Memory save: Mean velocity components (rarely used)
        float decay_const;                   // Legacy: Single nuclide decay constant (still used for compatibility)
        float conc;                          // Legacy: Single nuclide concentration (still used for output)
        float concentrations[N_NUCLIDES];  // Essential: Multi-nuclide concentration vector (60 nuclides)
        float age;                           // Essential: Particle age for decay calculations
        float virtual_distance;              // Essential: Used in dispersion calculations
        float u_wind, v_wind, w_wind;        // Essential: Meteorological wind components
        float sigma_z, sigma_h;              // Essential: Turbulent dispersion parameters
        float drydep_vel;                    // Essential: Dry deposition velocity
        float radi, prho;                    // Essential: Particle radius and density (used in settling)
        // float dsig;                      // Memory save: Size standard deviation (less critical)
        curandState* randState;              // Essential: Random number generator state

        int timeidx;                         // Essential: Time index for particle tracking
        int flag;                            // Essential: Particle active flag (heavily used)
        int dir;                             // Essential: Direction flag (used in reflection)
        int ensemble_id;                     // Essential: Ensemble member ID for EKI parallel execution (0~99)

        LDMpart() :
            x(0.0f), y(0.0f), z(0.0f),
            u(0.0f), v(0.0f), w(0.0f),
            up(0.0f), vp(0.0f), wp(0.0f),
            decay_const(0.0f),
            conc(0.0f),
            age(0.0f),
            virtual_distance(1e-5),
            u_wind(0.0f), v_wind(0.0f), w_wind(0.0f),
            sigma_z(0.0f), sigma_h(0.0f),
            drydep_vel(0.0f), radi(0.0f), prho(0.0f),
            randState(0),
            timeidx(0), flag(0), dir(1), ensemble_id(0){
                // Initialize multi-nuclide concentrations to zero
                for(int i = 0; i < N_NUCLIDES; i++) concentrations[i] = 0.0f;
            }

        LDMpart(float _x, float _y, float _z,
                float _decayConstant, float _concentration,
                float _drydep_vel, int _timeidx)  :
                x(_x), y(_y), z(_z),
                u(0.0f), v(0.0f), w(0.0f),
                up(0.0f), vp(0.0f), wp(0.0f),
                decay_const(_decayConstant),
                conc(_concentration),
                age(0.0f),
                virtual_distance(1e-5),
                u_wind(0.0f), v_wind(0.0f), w_wind(0.0f),
                sigma_z(0.0f), sigma_h(0.0f),
                drydep_vel(_drydep_vel), radi(0.0f), prho(0.0f),
                randState(0),
                timeidx(_timeidx), flag(0), dir(1), ensemble_id(0){
                    // Initialize multi-nuclide concentrations to zero
                    // (will be properly set in initialization functions)
                    for(int i = 0; i < N_NUCLIDES; i++) concentrations[i] = 0.0f;
                }

        LDMpart(float _x, float _y, float _z,
                float _decayConstant, float _concentration,
                float _drydep_vel, float radi, float prho, int _timeidx)  :
                x(_x), y(_y), z(_z),
                u(0.0f), v(0.0f), w(0.0f),
                up(0.0f), vp(0.0f), wp(0.0f),
                decay_const(_decayConstant),
                conc(_concentration),
                age(0.0f),
                virtual_distance(1e-5),
                u_wind(0.0f), v_wind(0.0f), w_wind(0.0f),
                sigma_z(0.0f), sigma_h(0.0f),
                drydep_vel(_drydep_vel), radi(radi), prho(prho),
                randState(0),
                timeidx(_timeidx), flag(0), dir(1), ensemble_id(0){
                    // Initialize multi-nuclide concentrations to zero
                    // (will be properly set in initialization functions)
                    for(int i = 0; i < N_NUCLIDES; i++) concentrations[i] = 0.0f;
                }

    };

    std::vector<LDMpart> part;
    LDMpart* d_part = nullptr;

    // ldm_func.cuh
    void printParticleData();
    void allocateGPUMemory();
    void runSimulation();
    void runSimulation_eki();
    void runSimulation_eki_dump();

    void findBoundingBox();
    void startTimer();   // Used in main
    void stopTimer();     // Used in main

    void createTextureObjects(); 
    void destroyTextureObjects();

    // ldm_init.cuh
    void loadSimulationConfiguration();  // Used in main (legacy)
    void cleanOutputDirectory();  // Used in loadSimulationConfiguration
    void calculateAverageSettlingVelocity();  // Used in main
    void initializeParticles();
    void initializeParticlesEKI();  // EKI mode particle initialization using true_emissions
    void initializeParticlesEKI_AllEnsembles(float* ensemble_states, int num_ensembles, int num_timesteps);  // EKI ensemble mode initialization
    void calculateSettlingVelocity();
    void loadReceptorConfig();  // receptor.conf parser (separated from EKI settings)
    void loadEKISettings();

    // Modernized config parsers (Phase 1 - INPUT_MODERNIZATION_PLAN.md)
    void loadSimulationConfig();   // simulation.conf parser
    void loadPhysicsConfig();       // physics.conf parser
    void loadSourceConfig();        // source.conf parser
    void loadNuclidesConfig();      // nuclides.conf parser
    void loadAdvancedConfig();      // advanced.conf parser

    // ldm_mdata.cuh
    void initializeFlexGFSData();  // Used in main
    void loadFlexGFSData();  // Used in time_update_mpi
    void loadMeteorologicalHeightData();
    void loadFlexHeightData();

    // ldm_plot.cuh
    int countActiveParticles();
    void swapByteOrder(float& value);
    void swapByteOrder(int& value);
    void outputParticlesBinaryMPI(int timestep);
    void outputParticlesBinaryMPI_ens(int timestep);

    // // Concentration tracking functions
    void log_first_particle_concentrations(int timestep, float currentTime);
    void log_all_particles_nuclide_ratios(int timestep, float currentTime);
    void log_first_particle_cram_detail(int timestep, float currentTime, float dt_used);
    void log_first_particle_decay_analysis(int timestep, float currentTime);
    
    // // Validation functions for CRAM4 reference data
    void exportValidationData(int timestep, float currentTime);
    void exportConcentrationGrid(int timestep, float currentTime);
    void exportNuclideTotal(int timestep, float currentTime);

    // EKI observation system functions
    void initializeEKIObservationSystem();
    void computeReceptorObservations(int timestep, float currentTime);
    void computeReceptorObservations_AllEnsembles(int timestep, float currentTime, int num_ensembles, int num_timesteps);
    void saveEKIObservationResults();
    void cleanupEKIObservationSystem();
    void resetEKIObservationSystemForNewIteration();

    // Grid receptor debug mode functions
    void initializeGridReceptors(int grid_count, float grid_spacing);
    void computeGridReceptorObservations(int timestep, float currentTime);
    void saveGridReceptorData();
    void cleanupGridReceptorSystem();

    // EKI IPC functions
    const EKIConfig& getEKIConfig() const { return g_eki; }
    const std::vector<std::vector<float>>& getEKIObservations() const { return eki_observations; }
    bool writeEKIObservationsToSharedMemory(void* writer);

    // ldm_cram2.cuh
    // static void gauss_solve_pivot_inplace(double* A, double* b, double* x, int N);
    // static void cram48_expm_times_ej_host(const std::vector<double>& A, int j, std::vector<double>& out_col);
    // bool build_T_on_host_and_upload(const char* A60_csv_path);
    // void debug_print_T_head(int k);

    bool load_A_csv(const char* path, std::vector<double>& A_out);
    void gauss_solve_inplace(std::vector<double>& M, std::vector<double>& b, int n);
    void cram48_expm_times_ej_host(const std::vector<double>& A, int j, std::vector<double>& col_out);
    bool build_T_matrix_and_upload(const char* A60_csv_path);
    bool initialize_cram_system(const char* A60_csv_path);

    // EKI용 기상자료 사전 로딩 함수들
    int calculateRequiredMeteoFiles();
    bool preloadAllEKIMeteorologicalData();
    bool loadSingleMeteoFile(int file_index, FlexPres*& pres_data, FlexUnis*& unis_data, std::vector<float>& hgt_data);
    void cleanupEKIMeteorologicalData();

    // NaN 디버깅 함수들
    void checkParticleNaN(const std::string& location, int max_check = 10);
    void checkMeteoDataNaN(const std::string& location);
};

#define LDM_CLASS_DECLARED 1

// Global nuclide count variable
extern int g_num_nuclides;

// Single-process constants (MPI removed)
// Previously: mpiRank = 1, mpiSize = 1
// Now using index 0 for all g_mpi array accesses
constexpr int PROCESS_INDEX = 0;

// Note: Core constants are now compile-time constants for CUDA compatibility

struct GridConfig {
    float start_lat{36.0f};
    float start_lon{140.0f};
    float end_lat{37.0f};
    float end_lon{141.0f};
    float lat_step{0.5f};
    float lon_step{0.5f};
};

// Grid configuration loader (implemented in ldm.cu)
GridConfig loadGridConfig();

// MPI variables removed - using single process mode
// g_mpi arrays now accessed with index 0 (PROCESS_INDEX)

// void loadRadionuclideData() {
//     std::vector<std::string> species_names = g_config.getStringArray("species_names");
//     std::vector<float> decay_constants = g_config.getFloatArray("decay_constants");
//     std::vector<float> deposition_velocities = g_config.getFloatArray("deposition_velocities");
//     std::vector<float> particle_sizes = g_config.getFloatArray("particle_sizes");
//     std::vector<float> particle_densities = g_config.getFloatArray("particle_densities");
//     std::vector<float> size_standard_deviations = g_config.getFloatArray("size_standard_deviations");
    
//     for (int i = 0; i < 4 && i < species_names.size(); i++) {
//         g_mpi.species[i] = species_names[i];
//         g_mpi.decayConstants[i] = (i < decay_constants.size()) ? decay_constants[i] : 1.00e-6f;
//         g_mpi.depositionVelocities[i] = (i < deposition_velocities.size()) ? deposition_velocities[i] : 0.01f;
//         g_mpi.particleSizes[i] = (i < particle_sizes.size()) ? particle_sizes[i] : 0.6f;
//         g_mpi.particleDensities[i] = (i < particle_densities.size()) ? particle_densities[i] : 2500.0f;
//         g_mpi.sizeStandardDeviations[i] = (i < size_standard_deviations.size()) ? size_standard_deviations[i] : 0.01f;
//     }
// }
// float __Z[720] = {0.0f, };


// Note: ldm_cram2.cuh moved to early include section (after ldm_config.cuh) to provide N_NUCLIDES macro
#include "../kernels/ldm_kernels.cuh"

// Refactored initialization modules
#include "../init/ldm_init_particles.cuh"
#include "../init/ldm_init_config.cuh"

// Refactored meteorological data modules
#include "../data/meteo/ldm_mdata_loading.cuh"
#include "../data/meteo/ldm_mdata_processing.cuh"
#include "../data/meteo/ldm_mdata_cache.cuh"

// Refactored simulation modules
#include "../simulation/ldm_func_simulation.cuh"
#include "../simulation/ldm_func_particle.cuh"
#include "../simulation/ldm_func_output.cuh"

// Refactored visualization modules
#include "../visualization/ldm_plot_vtk.cuh"
#include "../visualization/ldm_plot_utils.cuh"
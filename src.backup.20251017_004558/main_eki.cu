#include "core/ldm.cuh"
#include "physics/ldm_nuclides.cuh"
#include "ipc/ldm_eki_reader.cuh"
#include "ipc/ldm_eki_writer.cuh"
#include "debug/memory_doctor.cuh"
#include "debug/kernel_error_collector.cuh"
#include "simulation/ldm_func_output.cuh"
#include "colors.h"

// Standard library includes
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>

// Physics model global variables (will be loaded from setting.txt)
int g_num_nuclides;  // Default value, updated from nuclide config
int g_turb_switch;    // Default values, overwritten by setting.txt
int g_drydep;
int g_wetdep;
int g_raddecay;

// Log file handle (global for cross-compilation-unit access)
std::ofstream* g_log_file = nullptr;

int main(int argc, char** argv) {

    // Single process mode (MPI removed)

    // Clean logs directory and redirect output
    std::cout << "\n" << Color::BOLD << Color::CYAN
              << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
              << "  LDM-EKI: Source Term Inversion with Ensemble Kalman Methods\n"
              << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
              << Color::RESET << std::endl;

    // Create necessary directories
    std::cout << Color::CYAN << "[SYSTEM] " << Color::RESET << "Creating output directories..." << std::flush;
    system("mkdir -p logs output/plot_vtk_prior output/plot_vtk_ens output/results logs/eki_iterations 2>/dev/null");
    std::cout << " done\n";

    // Use centralized cleanup script for data cleanup
    std::cout << Color::CYAN << "[SYSTEM] " << Color::RESET << "Running cleanup script..." << std::endl;
    int cleanup_ret = system("python3 util/cleanup.py");

    if (cleanup_ret == 0) {
        std::cout << Color::GREEN << "✓ " << Color::RESET << "Cleanup completed\n" << std::endl;
    } else {
        std::cout << Color::YELLOW << "Cleanup skipped or failed (code: " << cleanup_ret << ")\n"
                  << Color::RESET << std::endl;
    }
    
    // Open log file for output redirection
    std::ofstream logFile("logs/ldm_eki_simulation.log");
    if (!logFile.is_open()) {
        std::cerr << Color::RED << "[ERROR] Could not open log file logs/ldm_eki_simulation.log"
                  << Color::RESET << std::endl;
        return 1;
    }
    
    // Redirect cout and cerr to log file while keeping console output
    std::streambuf* coutbuf = std::cout.rdbuf();
    std::streambuf* cerrbuf = std::cerr.rdbuf();

    // Color-stripping streambuf: removes ANSI escape codes before writing to log
    class ColorStripStreambuf : public std::streambuf {
        std::streambuf* dest;
        enum State { NORMAL, ESC, CSI };
        State state;
    public:
        ColorStripStreambuf(std::streambuf* d) : dest(d), state(NORMAL) {}

        int overflow(int c) {
            if (c == EOF) return !EOF;

            switch (state) {
                case NORMAL:
                    if (c == '\033') {  // ESC character
                        state = ESC;
                        return c;
                    }
                    return dest->sputc(c);

                case ESC:
                    if (c == '[') {  // CSI sequence
                        state = CSI;
                        return c;
                    }
                    // Not a CSI, output the ESC and this char
                    dest->sputc('\033');
                    state = NORMAL;
                    return dest->sputc(c);

                case CSI:
                    // Skip all characters until we find the final byte (0x40-0x7E)
                    if (c >= 0x40 && c <= 0x7E) {
                        state = NORMAL;
                    }
                    return c;  // Consume but don't output
            }
            return c;
        }

        int sync() {
            return dest->pubsync();
        }
    };

    // Tee streambuf: outputs to console and color-stripped log file
    class TeeStreambuf : public std::streambuf {
        std::streambuf* console;
        ColorStripStreambuf* logStrip;
    public:
        TeeStreambuf(std::streambuf* con, ColorStripStreambuf* log) : console(con), logStrip(log) {}
        int overflow(int c) {
            if (c == EOF) return !EOF;
            int r1 = console->sputc(c);
            int r2 = logStrip->overflow(c);
            return (r1 == EOF || r2 == EOF) ? EOF : c;
        }
        int sync() {
            int r1 = console->pubsync();
            int r2 = logStrip->sync();
            return (r1 == 0 && r2 == 0) ? 0 : -1;
        }
    };

    static ColorStripStreambuf log_strip_cout(logFile.rdbuf());
    static ColorStripStreambuf log_strip_cerr(logFile.rdbuf());
    static TeeStreambuf tee_cout(coutbuf, &log_strip_cout);
    static TeeStreambuf tee_cerr(cerrbuf, &log_strip_cerr);

    std::cout.rdbuf(&tee_cout);
    std::cerr.rdbuf(&tee_cerr);

    // Point global log file handle to this log file
    g_log_file = &logFile;

    // Test log-only stream immediately
    if (g_log_file && g_log_file->is_open()) {
        *g_log_file << "[TEST] g_log_file initialized successfully\n";
        g_log_file->flush();
    }

    // Test cross-compilation-unit access
    test_g_logonly_from_output_module();

    std::cout << Color::CYAN << "[SYSTEM] " << Color::RESET
              << "Logging to " << Color::BOLD << "logs/ldm_eki_simulation.log" << Color::RESET << "\n" << std::endl;

    // Log-only: detailed startup information
    if (g_log_file) {
        *g_log_file << "\n========================================\n";
        *g_log_file << "LDM-EKI Detailed Simulation Log\n";
        *g_log_file << "========================================\n";
        *g_log_file << "Start time: " << std::chrono::system_clock::now().time_since_epoch().count() << "\n";
        *g_log_file << "Working directory: " << getenv("PWD") << "\n";
        *g_log_file << "========================================\n\n";
    }

    // Load nuclide configuration
    NuclideConfig* nucConfig = NuclideConfig::getInstance();
    std::string nuclide_config_file = "./input/nuclides_config_1.txt";

    if (!nucConfig->loadFromFile(nuclide_config_file)) {
        std::cerr << Color::RED << "[ERROR] Failed to load nuclide configuration"
                  << Color::RESET << std::endl;
        return 1;
    }

    // Update global nuclide count
    g_num_nuclides = nucConfig->getNumNuclides();

    // Log-only: nuclide configuration details
    *g_log_file << "[CONFIG] Nuclide configuration loaded successfully\n";
    *g_log_file << "  Number of nuclides: " << g_num_nuclides << "\n";
    *g_log_file << "  Configuration file: " << nuclide_config_file << "\n\n";

    LDM ldm;

    // Modern configuration loading (Phase 2 integration)
    std::cout << "\n" << Color::BOLD << "Loading Configuration" << Color::RESET << "\n";

    ldm.loadSimulationConfig();      // simulation.conf
    ldm.loadPhysicsConfig();          // physics.conf
    ldm.loadSourceConfig();           // source.conf
    ldm.loadNuclidesConfig();         // nuclides.conf
    ldm.loadAdvancedConfig();         // advanced.conf

    std::cout << std::endl;

    // Log-only: simulation configuration details
    *g_log_file << "[CONFIG] Modernized configuration loaded\n";
    *g_log_file << "  Physics switches: TURB=" << g_turb_switch
            << " DRYDEP=" << g_drydep
            << " WETDEP=" << g_wetdep
            << " RADDECAY=" << g_raddecay << "\n\n";

    // Load EKI settings
    std::cout << "Loading configuration from " << Color::BOLD << "input/eki_settings.txt" << Color::RESET << "..." << std::flush;
    ldm.loadEKISettings();
    std::cout << " done\n";

    // Initialize Memory Doctor if enabled in settings
    extern MemoryDoctor g_memory_doctor;
    g_memory_doctor.setEnabled(ldm.getEKIConfig().memory_doctor_mode);

    // Initialize CRAM system with A60.csv matrix (moved from loadSimulationConfiguration)
    std::cout << "Initializing CRAM decay system..." << std::flush;
    if (!ldm.initialize_cram_system("./cram/A60.csv")) {
        std::cerr << Color::RED << "\n[ERROR] " << Color::RESET
                  << "CRAM system initialization failed" << std::endl;
        return 1;
    }
    std::cout << " done\n";

    // Clean output directory (moved from loadSimulationConfiguration)
    ldm.cleanOutputDirectory();

    std::cout << std::endl;

    ldm.calculateAverageSettlingVelocity();
    ldm.initializeParticlesEKI();

    // Preload all meteorological data for EKI mode
    if (!ldm.preloadAllEKIMeteorologicalData()) {
        std::cerr << Color::RED << "\n[ERROR] " << Color::RESET
                  << "Failed to preload meteorological data" << std::endl;
        return 1;
    }

    ldm.allocateGPUMemory();

    // NOTE: Don't initialize EKI observation system here for single mode
    // It will be initialized properly for ensemble mode in the iteration loop

    // Initialize EKI IPC Writer
    LDM_EKI_IPC::EKIWriter eki_writer;
    int ensemble_size = ldm.getEKIConfig().ensemble_size;
    int num_receptors = ldm.getEKIConfig().receptor_locations.size();
    
    // Calculate number of timesteps dynamically
    // time_end (seconds) / EKI_TIME_INTERVAL (minutes) = number of observations
    const EKIConfig& eki_config = ldm.getEKIConfig();
    float eki_interval_minutes = eki_config.time_interval;
    float eki_interval_seconds = eki_interval_minutes * 60.0f;
    int num_timesteps = (int)(time_end / eki_interval_seconds);

    // Initialize EKI writer with full configuration
    if (!eki_writer.initialize(eki_config, num_timesteps)) {
        std::cerr << Color::RED << "[ERROR] Failed to initialize EKI IPC Writer"
                  << Color::RESET << std::endl;
        return 1;
    }

    ldm.startTimer();

    // Initialize EKI observation system for single mode
    std::cout << "\nInitializing observation system for single mode..." << std::endl;
    ldm.initializeEKIObservationSystem();

    // Enable VTK output for single mode run
    ldm.enable_vtk_output = true;
    std::cout << Color::MAGENTA << "[VTK] " << Color::RESET << "Output enabled for single mode run" << std::endl;

    // Run EKI simulation with preloaded meteorological data
    std::cout << "\nRunning forward simulation..." << std::endl;

    // Enable kernel error collection
    KernelErrorCollector::enableCollection();

    ldm.runSimulation_eki();

    ldm.stopTimer();

    // Report any kernel errors that occurred during simulation
    KernelErrorCollector::reportAllErrors();
    KernelErrorCollector::disableCollection();

    std::cout << Color::GREEN << "Simulation completed successfully"
              << Color::RESET << "\n" << std::endl;

    // Save EKI observation results
    ldm.saveEKIObservationResults();

    // Write observations to shared memory
    std::cout << "\nWriting observations to shared memory..." << std::endl;
    
    // Get actual observation data from LDM object
    const std::vector<std::vector<float>>& observations = ldm.getEKIObservations();
    
    if (observations.empty()) {
        std::cerr << Color::RED << "[ERROR] No observations collected during simulation"
                  << Color::RESET << std::endl;
        return 1;
    }
    
    // Convert observations matrix to flat array for shared memory
    // observations[timestep][receptor] -> flat array[receptor * timesteps + timestep]
    std::vector<float> flat_observations(num_receptors * num_timesteps, 0.0f);
    
    for (size_t t = 0; t < observations.size() && t < static_cast<size_t>(num_timesteps); t++) {
        for (size_t r = 0; r < static_cast<size_t>(num_receptors) && r < observations[t].size(); r++) {
            flat_observations[r * num_timesteps + t] = observations[t][r];
        }
    }
    
    std::cout << "Collected " << observations.size() << " observation timesteps with "
              << (observations.empty() ? 0 : observations[0].size()) << " receptors each" << std::endl;

    bool success = eki_writer.writeObservations(flat_observations.data(), num_receptors, num_timesteps);

    if (!success) {
        std::cerr << Color::RED << "[ERROR] Failed to write observations to shared memory"
                  << Color::RESET << std::endl;
        return 1;
    }

    std::cout << Color::GREEN << "Observations successfully written to shared memory"
              << Color::RESET << "\n" << std::endl;

    // Launch Python EKI script in background
    std::cout << "Launching Python EKI script in background..." << std::endl;
    int ret = system("PYTHONPATH=src/eki:$PYTHONPATH python src/eki/RunEstimator.py input_config input_data > logs/python_eki_output.log 2>&1 &");
    if (ret != 0) {
        std::cout << Color::YELLOW << "Warning: Failed to launch Python script (code: " << ret << ")"
                  << Color::RESET << std::endl;
    } else {
        std::cout << Color::GREEN << "Python script launched successfully in background"
                  << Color::RESET << std::endl;
        std::cout << "Python output will be saved to logs/python_eki_output.log" << std::endl;
        // Give Python a moment to start up
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // ========================================================================
    // ITERATION LOOP: Process multiple rounds of ensemble states from Python
    // ========================================================================
    int max_iterations = ldm.getEKIConfig().iteration;
    int current_iteration = 0;

    std::cout << "\nMaximum iterations configured: " << max_iterations << std::endl;

    LDM_EKI_IPC::EKIReader eki_reader;
    bool continue_iterations = true;

    while (continue_iterations && current_iteration < max_iterations) {
        current_iteration++;

        // ========================================================================
        // Wait for ensemble states from Python
        // ========================================================================
        std::cout << "\n" << Color::BOLD << Color::CYAN
                  << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                  << "  ITERATION " << current_iteration << "/" << max_iterations;

        // Add FINAL emphasis for last iteration
        if (current_iteration == max_iterations) {
            std::cout << " (FINAL - VTK OUTPUT)";
        }

        std::cout << "\n"
                  << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                  << Color::RESET << std::endl;

        if (!eki_reader.waitForEnsembleData(60)) {  // 60 second timeout
            if (current_iteration == 1) {
                std::cerr << Color::RED << "\n[ERROR] " << Color::RESET
                          << "Timeout waiting for initial ensemble data\n";
                std::cerr << "  Python process may have crashed\n";
                std::cerr << "  Check logs/python_eki_output.log for details" << std::endl;
            } else {
                std::cout << Color::GREEN << "\n✓ " << Color::RESET
                          << "Python completed all iterations" << std::endl;
                continue_iterations = false;
            }
            break;
        }

        // Read ensemble states
        std::vector<float> ensemble_data;
        int num_states, num_ensemble;

        if (!eki_reader.readEnsembleStates(ensemble_data, num_states, num_ensemble)) {
            std::cerr << Color::RED << "[ERROR] Failed to read ensemble states from shared memory"
                      << Color::RESET << std::endl;
            continue_iterations = false;
            break;
        }

        // Log-only: detailed ensemble data received info
        *g_log_file << "\n[ITERATION " << current_iteration << "] Ensemble data received from Python\n";
        *g_log_file << "  Dimensions: " << num_states << " states × " << num_ensemble << " members\n";
        *g_log_file << "  Total values: " << ensemble_data.size() << "\n";
        *g_log_file << "  Memory size: " << (ensemble_data.size() * sizeof(float) / 1024.0) << " KB\n";

        // Calculate statistics
        float min_val = *std::min_element(ensemble_data.begin(), ensemble_data.end());
        float max_val = *std::max_element(ensemble_data.begin(), ensemble_data.end());
        float sum = std::accumulate(ensemble_data.begin(), ensemble_data.end(), 0.0f);
        float mean_val = sum / ensemble_data.size();

        std::cout << Color::BOLD << "\nEnsemble State Summary\n" << Color::RESET;
        std::cout << "  Dimensions    : " << num_states << " states × " << num_ensemble << " members\n";
        std::cout << "  Min value     : " << std::scientific << std::setprecision(2) << min_val << "\n";
        std::cout << "  Max value     : " << max_val << "\n";
        std::cout << "  Mean value    : " << mean_val << std::endl;

#ifdef DEBUG
        // Count zeros and negative values for validation
        int zero_count = 0;
        int negative_count = 0;
        int tiny_count = 0;  // Values < 1e6
        for (const auto& val : ensemble_data) {
            if (val == 0.0f) zero_count++;
            if (val < 0.0f) negative_count++;
            if (val > 0.0f && val < 1.0e6f) tiny_count++;
        }

        std::cout << Color::YELLOW << "[DEBUG] " << Color::RESET << "Value analysis:" << std::endl;
        std::cout << "  Zero values: " << zero_count << " / " << ensemble_data.size()
                  << " (" << (100.0f * zero_count / ensemble_data.size()) << "%)" << std::endl;
        std::cout << "  Negative values: " << negative_count << std::endl;
        std::cout << "  Tiny values (<1e6): " << tiny_count << std::endl;

        // Highlight if negatives are received
        if (negative_count > 0) {
            std::cout << Color::RED << "  Negative values detected in iteration "
                      << current_iteration << Color::RESET << std::endl;
        } else {
            std::cout << Color::GREEN << "  ✓ All values non-negative\n"
                      << Color::RESET;
        }
#endif

#ifdef DEBUG_VERBOSE
        // Save what LDM receives for comparison
        {
            std::string debug_filename = "/tmp/eki_debug/ldm_receives_iter_" + std::to_string(current_iteration) + ".txt";
            std::ofstream debug_file(debug_filename);
            if (debug_file.is_open()) {
                debug_file << "Iteration " << current_iteration << " data received by LDM\n";
                debug_file << "Shape: " << num_states << " x " << num_ensemble << "\n";
                debug_file << "Min: " << min_val << ", Max: " << max_val << ", Mean: " << mean_val << "\n";
                debug_file << "Negative count: " << negative_count << "\n";
                debug_file << "First 10 values: ";
                for (int idx = 0; idx < std::min(10, (int)ensemble_data.size()); idx++) {
                    debug_file << ensemble_data[idx] << " ";
                }
                debug_file << "\n";
                debug_file.close();
                std::cout << Color::YELLOW << "[DEBUG] " << Color::RESET
                          << "Saved data to " << debug_filename << std::endl;
            }
        }

        // Compare with previous iteration if available
        static float prev_min = 0.0f, prev_max = 0.0f, prev_mean = 0.0f;
        if (current_iteration > 1) {
            std::cout << Color::YELLOW << "[DEBUG] " << Color::RESET << "Change from iteration "
                      << (current_iteration - 1) << ":" << std::endl;
            std::cout << "  Min:  " << ((min_val - prev_min) / prev_min * 100.0f) << "%" << std::endl;
            std::cout << "  Max:  " << ((max_val - prev_max) / prev_max * 100.0f) << "%" << std::endl;
            std::cout << "  Mean: " << ((mean_val - prev_mean) / prev_mean * 100.0f) << "%" << std::endl;
        }
        prev_min = min_val;
        prev_max = max_val;
        prev_mean = mean_val;
#endif

#ifdef DEBUG_VERBOSE
        // Display detailed data for first iteration only
        if (current_iteration == 1) {
            std::cout << Color::YELLOW << "\n[DEBUG] " << Color::RESET
                      << "Sample data (first state, first 20 ensemble members):" << std::endl;
            int display_count = std::min(20, num_ensemble);
            for (int i = 0; i < display_count; i++) {
                std::cout << "  [state 0, ensemble " << i << "] = " << ensemble_data[i] << std::endl;
            }

            std::cout << Color::YELLOW << "\n[DEBUG] " << Color::RESET
                      << "Sample data (first ensemble member, first 10 states):" << std::endl;
            int state_display = std::min(10, num_states);
            for (int s = 0; s < state_display; s++) {
                std::cout << "  [state " << s << ", ensemble 0] = "
                          << ensemble_data[s * num_ensemble] << std::endl;
            }
        }
#endif

        // ========================================================================
        // Ensemble Mode: Initialize particles with ensemble states
        // ========================================================================
        std::cout << "\n" << Color::YELLOW << "[ENSEMBLE] " << Color::RESET
                  << "Preparing ensemble simulation for iteration " << current_iteration << std::endl;

        // Data format conversion: Python sends [state0_ens0, state0_ens1, ..., state1_ens0, ...]
        // We need: [ens0_state0, ens0_state1, ..., ens1_state0, ...] for row-major ensemble matrix
        std::vector<float> ensemble_matrix(num_ensemble * num_states);

        for (int s = 0; s < num_states; s++) {
            for (int e = 0; e < num_ensemble; e++) {
                // Python: ensemble_data[s * num_ensemble + e]
                // LDM needs: ensemble_matrix[e * num_states + s]
                ensemble_matrix[e * num_states + s] = ensemble_data[s * num_ensemble + e];
            }
        }

#ifdef DEBUG_VERBOSE
        // Check for zero values in ensemble matrix (first iteration only)
        if (current_iteration == 1) {
            int zero_count = 0;
            int nonzero_count = 0;
            float min_val = 1e20f, max_val = -1e20f;
            std::vector<std::string> zero_locations;

            std::cout << Color::YELLOW << "[DEBUG] " << Color::RESET
                      << "Checking full matrix: " << num_ensemble << " × " << num_states << "..." << std::endl;

            for (int e = 0; e < num_ensemble; e++) {
                for (int s = 0; s < num_states; s++) {
                    float val = ensemble_matrix[e * num_states + s];
                    if (val == 0.0f) {
                        zero_count++;
                        if (zero_locations.size() < 50) {
                            zero_locations.push_back("[E" + std::to_string(e) + ",S" + std::to_string(s) + "]");
                        }
                    } else {
                        nonzero_count++;
                        min_val = std::min(min_val, val);
                        max_val = std::max(max_val, val);
                    }
                }
            }

            std::cout << "  Total values: " << (num_ensemble * num_states) << std::endl;
            std::cout << "  Zero values: " << zero_count
                      << " (" << (100.0f * zero_count / (num_ensemble * num_states)) << "%)" << std::endl;
            std::cout << "  Non-zero values: " << nonzero_count << std::endl;

            if (nonzero_count > 0) {
                std::cout << "  Min (non-zero): " << min_val << std::endl;
                std::cout << "  Max: " << max_val << std::endl;
            }

            if (zero_count > 0) {
                std::cout << Color::YELLOW << "  Zero values detected in ensemble matrix!"
                          << Color::RESET << std::endl;
            } else {
                std::cout << Color::GREEN << "  ✓ All values non-zero\n" << Color::RESET;
            }
        }
#endif

        // Set ensemble mode flags (only on first iteration)
        if (current_iteration == 1) {
            ldm.is_ensemble_mode = true;
            ldm.ensemble_size = num_ensemble;
            ldm.ensemble_num_states = num_timesteps;  // Use observation timesteps (48) not state timesteps (24)

            // Select ensemble 7 for VTK output (fixed, not random)
            ldm.selected_ensemble_ids.clear();
            ldm.selected_ensemble_ids.push_back(7);

            std::cout << Color::YELLOW << "[ENSEMBLE] " << Color::RESET
                      << "Mode configured: " << num_ensemble << " ensembles, "
                      << num_timesteps << " observation timesteps" << std::endl;
            std::cout << Color::YELLOW << "[ENSEMBLE] " << Color::RESET
                      << "Selected ensemble 7 for VTK output (fixed)" << std::endl;

            // First time: cleanup single mode and initialize ensemble mode observation system
            ldm.cleanupEKIObservationSystem();
            ldm.initializeEKIObservationSystem();
        }

        // Enable VTK output ONLY on the final iteration for performance
        if (current_iteration == max_iterations) {
            ldm.enable_vtk_output = true;
            std::cout << Color::MAGENTA << "[VTK] " << Color::RESET
                      << "Output enabled for final iteration " << current_iteration << std::endl;
            std::cout << Color::MAGENTA << "[VTK] " << Color::RESET
                      << "Ensemble output parallelization: " << Color::BOLD << "50" << Color::RESET << " threads\n";
            std::cout << Color::MAGENTA << "[VTK] " << Color::RESET
                      << "Selected ensemble for output: " << Color::BOLD
                      << ldm.selected_ensemble_ids[0] << Color::RESET << std::endl;
        } else {
            ldm.enable_vtk_output = false;
            std::cout << Color::MAGENTA << "[VTK] " << Color::RESET
                      << "Output disabled for iteration " << current_iteration << " (performance optimization)" << std::endl;
        }

        // Clear previous particles for reinitialization
        ldm.part.clear();

        // Initialize particles for all ensembles with new states
        ldm.initializeParticlesEKI_AllEnsembles(ensemble_matrix.data(), num_ensemble, num_states);

#ifdef DEBUG_VERBOSE
        // Check concentrations array after initialization
        std::cout << Color::YELLOW << "[DEBUG] " << Color::RESET
                  << "Checking particle concentrations after initialization..." << std::endl;
        int check_count = 0;
        int particle_zero_count = 0;
        int particle_nonzero_count = 0;
        for (size_t i = 0; i < std::min(size_t(1000), ldm.part.size()); i++) {
            float total_conc = 0.0f;
            for (int nuc = 0; nuc < 60; nuc++) {
                total_conc += ldm.part[i].concentrations[nuc];
            }
            if (total_conc == 0.0f) {
                particle_zero_count++;
            } else {
                particle_nonzero_count++;
            }
            if (check_count < 5) {
                std::cout << "  Particle " << i << " (ens=" << ldm.part[i].ensemble_id
                          << ", timeidx=" << ldm.part[i].timeidx
                          << "): conc=" << ldm.part[i].conc
                          << ", sum=" << total_conc << std::endl;
                check_count++;
            }
        }
        std::cout << "  First 1000 particles: " << particle_nonzero_count << " non-zero, "
                  << particle_zero_count << " zero concentrations" << std::endl;
#endif

        // Verify particle count after initialization
        if (current_iteration == 1) {
            size_t expected_particles = static_cast<size_t>(num_ensemble) * num_states * (10000 / 24);
            std::cout << Color::YELLOW << "[ENSEMBLE] " << Color::RESET
                      << "Total particles after initialization: " << ldm.part.size() << std::endl;
            std::cout << Color::YELLOW << "[ENSEMBLE] " << Color::RESET
                      << "Expected particles: ~" << expected_particles
                      << " (" << num_ensemble << " ensembles × " << num_states
                      << " states × " << (10000/24) << " particles/state)" << std::endl;
        } else {
            std::cout << Color::YELLOW << "[ENSEMBLE] " << Color::RESET
                      << "Particles reinitialized: " << ldm.part.size() << std::endl;
        }

        if (ldm.part.size() == 0) {
            std::cerr << Color::RED << "[ERROR] No particles initialized! Check initializeParticlesEKI_AllEnsembles()"
                      << Color::RESET << std::endl;
            continue_iterations = false;
            break;
        }

        // Reallocate GPU memory for new particles
        if (ldm.d_part) {
            // IMPORTANT: Ensure all GPU operations are complete before freeing memory
            cudaDeviceSynchronize();
            cudaFree(ldm.d_part);
            ldm.d_part = nullptr;
        }
        ldm.allocateGPUMemory();  // Will allocate for new part.size()

        // Reset EKI observation system for this iteration
        // NOTE: Don't call cleanupEKIObservationSystem() here as it deallocates GPU memory
        // Instead, just reset the memory and host storage
        ldm.resetEKIObservationSystemForNewIteration();

    // Run ensemble simulation
    std::cout << "\n" << Color::YELLOW << "[ENSEMBLE] " << Color::RESET
              << "Starting forward simulation..." << std::endl;

    // Enable kernel error collection
    KernelErrorCollector::clearErrors();  // Clear errors from previous iteration
    KernelErrorCollector::enableCollection();

    ldm.startTimer();
    ldm.runSimulation_eki();
    ldm.stopTimer();

    // Report any kernel errors that occurred during ensemble simulation
    KernelErrorCollector::reportAllErrors();
    KernelErrorCollector::disableCollection();

    std::cout << Color::GREEN << "[ENSEMBLE] Simulation completed" << Color::RESET << "\n" << std::endl;

    // ========================================================================
    // Send ensemble observations back to Python
    // ========================================================================
    std::cout << Color::YELLOW << "[ENSEMBLE] " << Color::RESET
              << "Preparing to send observations to Python..." << std::endl;

    // Format: [num_ensemble × num_receptors × num_timesteps]
    // Python expects: flat array that can be reshaped to (num_ensemble, num_receptors, num_timesteps)

    // Get references to EKI data (eki_config already declared at line 138)
    auto& ensemble_observations = ldm.getEKIEnsembleObservations();

    // Flatten the 3D vector to a 1D array
    int total_obs_elements = num_ensemble * eki_config.num_receptors * num_timesteps;
    std::vector<float> flat_ensemble_observations(total_obs_elements);

    int idx = 0;
    for (int ens = 0; ens < num_ensemble; ens++) {
        // IMPORTANT: Match reference implementation order - timestep-major flattening
        // For each ensemble, we want: [T0_R0...R2, T1_R0...R2, ..., T23_R0...R2]
        for (int t = 0; t < num_timesteps; t++) {                  // timestep first (outer loop)
            for (int r = 0; r < eki_config.num_receptors; r++) {   // receptor second (inner loop)
                // eki_ensemble_observations is [ensemble][timestep][receptor]
                // We access as [ens][t][r] and flatten in timestep-major order
                flat_ensemble_observations[idx++] = ensemble_observations[ens][t][r];
            }
        }
    }

        // Initialize ensemble observation shared memory
        if (!eki_writer.initializeEnsembleObservations(num_ensemble, eki_config.num_receptors, num_timesteps)) {
            std::cerr << Color::RED << "[ERROR] Failed to initialize ensemble observation shared memory"
                      << Color::RESET << std::endl;
            continue_iterations = false;
            break;
        }

        // Write ensemble observations to shared memory
        if (!eki_writer.writeEnsembleObservations(flat_ensemble_observations.data(), num_ensemble, eki_config.num_receptors, num_timesteps, current_iteration)) {
            std::cerr << Color::RED << "[ERROR] Failed to write ensemble observations to shared memory"
                      << Color::RESET << std::endl;
            continue_iterations = false;
            break;
        }

    std::cout << Color::CYAN << "[IPC] " << Color::RESET << "Sent observations to Python: "
              << Color::BOLD << "[" << num_ensemble << " × " << eki_config.num_receptors
              << " × " << num_timesteps << "]" << Color::RESET
              << " (" << total_obs_elements << " values)" << std::endl;

    std::cout << "\n" << Color::BOLD << Color::ORANGE << "✓ Iteration " << current_iteration
              << "/" << max_iterations << " completed" << Color::RESET << "\n" << std::endl;

        // Log-only: iteration summary
        *g_log_file << "[ITERATION " << current_iteration << "] Completed successfully\n";
        *g_log_file << "  Observations sent: " << total_obs_elements << " values\n";
        *g_log_file << "  Observation range: [" << *std::min_element(flat_ensemble_observations.begin(), flat_ensemble_observations.end())
                << ", " << *std::max_element(flat_ensemble_observations.begin(), flat_ensemble_observations.end()) << "]\n";
        *g_log_file << "----------------------------------------\n\n";

    } // End of iteration loop

    // ========================================================================
    // Cleanup (after all iterations)
    // ========================================================================
    std::cout << "\n" << Color::BOLD << Color::CYAN
              << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
              << "  CLEANUP\n"
              << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
              << Color::RESET;

    std::cout << Color::GREEN << "✓ " << Color::RESET
              << "Completed " << Color::BOLD << current_iteration << Color::RESET
              << " iterations\n" << std::endl;

    // Cleanup EKI observation system
    std::cout << Color::CYAN << "[SYSTEM] " << Color::RESET << "Releasing resources..." << std::flush;
    ldm.cleanupEKIObservationSystem();
    ldm.cleanupEKIMeteorologicalData();

    // Cleanup shared memory
    eki_writer.cleanup();
    LDM_EKI_IPC::EKIWriter::unlinkSharedMemory();
    LDM_EKI_IPC::EKIReader::unlinkEnsembleSharedMemory();
    std::cout << " done\n";

    // Restore original stream buffers
    std::cout.rdbuf(coutbuf);
    std::cerr.rdbuf(cerrbuf);
    logFile.close();

    std::cout << "\n" << Color::ORANGE << Color::BOLD << "✓ Simulation completed" << Color::RESET << std::endl;
    std::cout << "  Logs: " << Color::BOLD << "logs/ldm_eki_simulation.log" << Color::RESET << std::endl;

    // ========================================================================
    // Automatic Post-Processing: Generate Visualization
    // ========================================================================
    std::cout << "\n" << Color::CYAN << "[VISUALIZATION] " << Color::RESET
              << "Generating comparison graphs..." << std::flush;

    // Check if visualization script exists
    std::ifstream viz_script("util/compare_all_receptors.py");
    if (viz_script.good()) {
        viz_script.close();

        int viz_ret = system("python3 util/compare_all_receptors.py > /tmp/ldm_viz.log 2>&1");

        if (viz_ret == 0) {
            std::cout << " done\n";
            std::cout << "  Output: " << Color::BOLD
                      << "output/results/all_receptors_comparison.png" << Color::RESET << std::endl;
        } else {
            std::cout << Color::RED << " failed\n" << Color::RESET;
            std::cout << Color::YELLOW << "  Visualization failed (code: " << viz_ret << ")\n"
                      << Color::RESET;
            std::cout << "  Check /tmp/ldm_viz.log for details\n";
            std::cout << "  Run manually: python3 util/compare_all_receptors.py" << std::endl;
        }
    } else {
        std::cout << Color::YELLOW << " skipped (script not found)\n" << Color::RESET;
    }

    return 0;
}
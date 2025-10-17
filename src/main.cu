/******************************************************************************
 * @file main.cu
 * @brief Standard simulation entry point (non-EKI mode)
 *
 * This entry point runs the LDM forward model in standalone mode without
 * Ensemble Kalman Inversion. Used for:
 * - Single forward simulations with known source terms
 * - Model validation and verification
 * - Performance benchmarking
 * - Testing new physics modules
 *
 * Key Differences from main_eki.cu:
 * - No IPC communication or Python coupling
 * - Single simulation run only (no iterations)
 * - Simpler initialization sequence
 * - Direct VTK output without performance constraints
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/

#include "core/ldm.cuh"
#include "physics/ldm_nuclides.cuh"

// Standard library includes
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <cmath>

// ===========================================================================
// Global Variables
// ===========================================================================

// Physics model switches (loaded from physics.conf)
int g_num_nuclides = 1;  // Number of nuclides in decay chain (default: 1)
int g_turb_switch = 0;   // Turbulent diffusion enable flag (0=off, 1=on)
int g_drydep = 0;        // Dry deposition enable flag (0=off, 1=on)
int g_wetdep = 0;        // Wet deposition enable flag (0=off, 1=on)
int g_raddecay = 0;      // Radioactive decay enable flag (0=off, 1=on)

/******************************************************************************
 * @brief Main execution function for standard simulation mode
 *
 * Executes a single forward simulation without EKI coupling:
 * 1. Load nuclide configuration and simulation settings
 * 2. Initialize CRAM decay system for radioactive decay chains
 * 3. Calculate particle settling velocities
 * 4. Initialize particles with source term
 * 5. Load and prepare meteorological data
 * 6. Allocate GPU memory for particle arrays
 * 7. Run forward simulation with VTK output
 *
 * @param[in] argc Command line argument count (currently unused)
 * @param[in] argv Command line arguments (currently unused)
 *
 * @return 0 on success, 1 on error
 *
 * @note Legacy MPI variables (mpiRank, mpiSize) maintained for compatibility
 * @note Uses modernized configuration system (simulation.conf, physics.conf, etc.)
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
int main(int argc, char** argv) {

    // ===========================================================================
    // Legacy MPI Compatibility (Single Process Mode)
    // ===========================================================================
    // These variables maintained for backward compatibility with legacy code
    // In single-process mode, mpiRank=1 and mpiSize=1 by convention
    mpiRank = 1;
    mpiSize = 1;

    // ===========================================================================
    // Nuclide Configuration Loading
    // ===========================================================================
    NuclideConfig* nucConfig = NuclideConfig::getInstance();
    std::string nuclide_config_file = "./input/nuclides_config_1.txt";

    if (!nucConfig->loadFromFile(nuclide_config_file)) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET
                  << "Failed to load nuclide configuration" << std::endl;
        return 1;
    }

    // Store nuclide count in global variable for kernel access
    g_num_nuclides = nucConfig->getNumNuclides();

    // ===========================================================================
    // LDM Initialization and Configuration
    // ===========================================================================
    LDM ldm;

    // Load legacy configuration format (simulation.txt, source.txt, etc.)
    ldm.loadSimulationConfiguration();

    // Initialize CRAM radioactive decay system with transition matrix
    std::cout << "[DEBUG] Initializing CRAM system..." << std::endl;
    if (!ldm.initialize_cram_system("./cram/A60.csv")) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET
                  << "CRAM system initialization failed" << std::endl;
        return 1;
    }
    std::cout << "[DEBUG] CRAM system initialization completed" << std::endl;

    // ===========================================================================
    // Particle and Meteorology Initialization
    // ===========================================================================

    // Calculate settling velocity for particles
    ldm.calculateAverageSettlingVelocity();

    // Initialize particle array with source term
    ldm.initializeParticles();

    // Load meteorological data (order is critical: height data first)
    ldm.loadFlexHeightData();    // Load vertical height levels
    ldm.initializeFlexGFSData(); // Load GFS data and calculate density

    // Allocate GPU memory for particles and meteorological fields
    ldm.allocateGPUMemory();

    // ===========================================================================
    // Run Forward Simulation
    // ===========================================================================
    ldm.startTimer();
    ldm.runSimulation();
    ldm.stopTimer();

    // MPI_Finalize();
    return 0;
}
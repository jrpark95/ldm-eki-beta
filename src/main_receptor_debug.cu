/******************************************************************************
 * @file main_receptor_debug.cu
 * @brief Grid receptor debug tool for spatial dose distribution analysis
 *
 * This diagnostic tool creates a regular grid of virtual receptors around the
 * source location to analyze the spatial distribution of particle deposition
 * and dose accumulation. Useful for:
 * - Validating dispersion model plume shape
 * - Identifying hotspots in dose distribution
 * - Debugging particle transport and deposition physics
 * - Verifying meteorological data influence
 *
 * Grid Configuration:
 * - Receptors arranged in (2*N+1) × (2*N+1) square grid
 * - Centered on source location
 * - Spacing specified in degrees (lat/lon)
 * - Example: N=5, spacing=0.1° → 11×11 = 121 receptors
 *
 * Output Files (in grid_receptors/ directory):
 * - receptor_locations.csv: Lat/lon coordinates of each receptor
 * - receptor_timeseries.csv: Dose accumulation over time
 * - receptor_summary.csv: Peak values and total deposition
 * - grid_heatmap.png: Spatial visualization (auto-generated)
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
 * @brief Main execution function for grid receptor debug mode
 *
 * Creates a grid of virtual receptors and runs a forward simulation to collect
 * spatial dose distribution data. Automatically generates visualization plots.
 *
 * Workflow:
 * 1. Parse command line arguments (grid_count, grid_spacing)
 * 2. Validate parameters
 * 3. Initialize LDM with grid receptor mode enabled
 * 4. Run simulation collecting data at each receptor
 * 5. Save CSV output files
 * 6. Generate visualization plots via Python script
 *
 * @param[in] argc Command line argument count (must be 3)
 * @param[in] argv Command line arguments:
 *                 argv[1] = grid_count (receptors in each direction, 1-20)
 *                 argv[2] = grid_spacing (degrees, 0.0-1.0)
 *
 * @return 0 on success, 1 on error
 *
 * @note Grid dimensions: (2*grid_count+1) × (2*grid_count+1) receptors
 * @note Large grids (grid_count > 10) may cause significant performance overhead
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
int main(int argc, char** argv) {
    std::cout << "=== LDM Grid Receptor Debug Tool ===" << std::endl;

    // ===========================================================================
    // Command Line Argument Parsing
    // ===========================================================================
    if (argc != 3) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET
                  << "Usage: " << argv[0] << " <grid_count> <grid_spacing>" << std::endl;
        std::cerr << "\nArguments:" << std::endl;
        std::cerr << "  grid_count   : Number of receptors in each direction from source (1-20)" << std::endl;
        std::cerr << "                 Example: 5 creates 11x11 grid (2*5+1 in each direction)" << std::endl;
        std::cerr << "  grid_spacing : Distance between receptors in degrees (0.0-1.0)" << std::endl;
        std::cerr << "                 Example: 0.1 creates 0.1° spacing (~11 km)" << std::endl;
        std::cerr << "\nExample Usage:" << std::endl;
        std::cerr << "  " << argv[0] << " 5 0.1" << std::endl;
        std::cerr << "  This creates (2*5+1)×(2*5+1) = 121 receptors in a square grid" << std::endl;
        return 1;
    }

    int grid_count = std::atoi(argv[1]);
    float grid_spacing = std::atof(argv[2]);

    // Validate input parameters
    if (grid_count <= 0 || grid_count > 20) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET
                  << "grid_count must be between 1 and 20 (got " << grid_count << ")" << std::endl;
        std::cerr << "  Smaller grids (1-5) recommended for performance" << std::endl;
        return 1;
    }

    if (grid_spacing <= 0.0f || grid_spacing > 1.0f) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET
                  << "grid_spacing must be between 0.0 and 1.0 degrees (got " << grid_spacing << ")" << std::endl;
        std::cerr << "  Typical values: 0.05° (fine), 0.1° (medium), 0.25° (coarse)" << std::endl;
        return 1;
    }

    // Calculate and display grid dimensions
    int total_receptors = (2 * grid_count + 1) * (2 * grid_count + 1);
    std::cout << "[INFO] Grid configuration:" << std::endl;
    std::cout << "  Grid count    : " << grid_count << " (in each direction from source)" << std::endl;
    std::cout << "  Grid spacing  : " << grid_spacing << " degrees (~"
              << (int)(grid_spacing * 111.0) << " km)" << std::endl;
    std::cout << "  Grid dimension: " << (2*grid_count+1) << "×" << (2*grid_count+1) << std::endl;
    std::cout << "  Total receptors: " << total_receptors << std::endl;

    // Create output directory for grid receptor data
    system("mkdir -p grid_receptors");

    // ===========================================================================
    // Legacy MPI Compatibility
    // ===========================================================================
    mpiRank = 1;
    mpiSize = 1;

    // ===========================================================================
    // Configuration Loading
    // ===========================================================================

    // Load nuclide decay chain configuration
    NuclideConfig* nucConfig = NuclideConfig::getInstance();
    std::string nuclide_config_file = "./cram/nuclides_config_single.txt";

    if (!nucConfig->loadFromFile(nuclide_config_file)) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET
                  << "Failed to load nuclide configuration" << std::endl;
        return 1;
    }

    // Store nuclide count in global variable
    g_num_nuclides = nucConfig->getNumNuclides();

    // ===========================================================================
    // LDM Initialization with Grid Receptor Mode
    // ===========================================================================
    LDM ldm;

    // Enable grid receptor debug mode with specified parameters
    ldm.is_grid_receptor_mode = true;
    ldm.grid_count = grid_count;
    ldm.grid_spacing = grid_spacing;

    // Load simulation configuration
    ldm.loadSimulationConfiguration();

    // Initialize CRAM radioactive decay system
    std::cout << "[DEBUG] Initializing CRAM system..." << std::endl;
    if (!ldm.initialize_cram_system("./cram/A60.csv")) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET
                  << "CRAM system initialization failed" << std::endl;
        return 1;
    }
    std::cout << "[DEBUG] CRAM system initialization completed" << std::endl;

    // ===========================================================================
    // Particle and Meteorology Setup
    // ===========================================================================

    // Calculate settling velocity and initialize particles
    ldm.calculateAverageSettlingVelocity();
    ldm.initializeParticles();

    // Load meteorological data (order is critical)
    ldm.loadFlexHeightData();    // Load vertical levels first
    ldm.initializeFlexGFSData(); // Then load wind/temperature fields

    // Initialize grid of virtual receptors centered on source
    std::cout << "[GRID] Initializing " << total_receptors << " grid receptors..." << std::endl;
    ldm.initializeGridReceptors(grid_count, grid_spacing);

    // Allocate GPU memory for particles and fields
    ldm.allocateGPUMemory();

    // ===========================================================================
    // Run Simulation with Grid Receptor Collection
    // ===========================================================================
    std::cout << "[INFO] Starting simulation..." << std::endl;
    ldm.startTimer();
    ldm.runSimulation();
    ldm.stopTimer();

    // ===========================================================================
    // Save Results
    // ===========================================================================

    // Export grid receptor time series and spatial data
    std::cout << "[GRID] Saving receptor data to CSV files..." << std::endl;
    ldm.saveGridReceptorData();

    std::cout << "[INFO] Grid receptor debug simulation completed successfully!" << std::endl;
    std::cout << "[INFO] Output files saved to: grid_receptors/" << std::endl;

    // Automatically run visualization script
    std::cout << "\n[VISUALIZATION] Generating analysis plots..." << std::endl;
    int viz_ret = system("python visualize_grid_receptors.py > grid_receptors/visualization.log 2>&1");

    if (viz_ret == 0) {
        std::cout << "[VISUALIZATION] Plots generated successfully!" << std::endl;
        std::cout << "\nGenerated files:" << std::endl;
        std::cout << "  - grid_receptors/grid_heatmap.png              (Spatial dose/particle distribution)" << std::endl;
        std::cout << "  - grid_receptors/all_receptors_overview.png    (All receptors time series)" << std::endl;
        std::cout << "  - grid_receptors/timeseries_top_receptors.png  (Top 16 detailed plots)" << std::endl;
        std::cout << "  - grid_receptors/summary_statistics.txt        (Statistics summary)" << std::endl;
        std::cout << "\nRecommended viewing order:" << std::endl;
        std::cout << "  1. cat grid_receptors/summary_statistics.txt" << std::endl;
        std::cout << "  2. open grid_receptors/all_receptors_overview.png" << std::endl;
        std::cout << "  3. open grid_receptors/grid_heatmap.png" << std::endl;
    } else {
        std::cerr << "[WARNING] Visualization script failed (exit code: " << viz_ret << ")" << std::endl;
        std::cerr << "  Check grid_receptors/visualization.log for details" << std::endl;
        std::cerr << "  You can run manually: python visualize_grid_receptors.py" << std::endl;
    }

    return 0;
}

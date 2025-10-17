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

// Physics model global variables (will be loaded from setting.txt)
int g_num_nuclides = 1;  // Default value, updated from nuclide config
int g_turb_switch = 0;    // Default values, overwritten by setting.txt
int g_drydep = 0;
int g_wetdep = 0;
int g_raddecay = 0;

int main(int argc, char** argv) {
    std::cout << "=== LDM Grid Receptor Debug Tool ===" << std::endl;

    // Parse command line arguments
    if (argc != 3) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Usage: " << argv[0] << " <grid_count> <grid_spacing>" << std::endl;
        std::cerr << "  grid_count: Number of receptors in each direction from source (e.g., 5)" << std::endl;
        std::cerr << "  grid_spacing: Distance between receptors in degrees (e.g., 0.1)" << std::endl;
        std::cerr << "  Example: " << argv[0] << " 5 0.1" << std::endl;
        std::cerr << "  This creates (2*5+1)×(2*5+1) = 121 receptors in a square grid" << std::endl;
        return 1;
    }

    int grid_count = std::atoi(argv[1]);
    float grid_spacing = std::atof(argv[2]);

    // Validate arguments
    if (grid_count <= 0 || grid_count > 20) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "grid_count must be between 1 and 20 (got " << grid_count << ")" << std::endl;
        return 1;
    }

    if (grid_spacing <= 0.0f || grid_spacing > 1.0f) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "grid_spacing must be between 0.0 and 1.0 degrees (got " << grid_spacing << ")" << std::endl;
        return 1;
    }

    int total_receptors = (2 * grid_count + 1) * (2 * grid_count + 1);
    std::cout << "[INFO] Grid configuration:" << std::endl;
    std::cout << "  Grid count: " << grid_count << " (in each direction)" << std::endl;
    std::cout << "  Grid spacing: " << grid_spacing << " degrees" << std::endl;
    std::cout << "  Total receptors: " << total_receptors << " (" << (2*grid_count+1) << "×" << (2*grid_count+1) << ")" << std::endl;

    // Create output directory
    system("mkdir -p grid_receptors");

    mpiRank = 1;
    mpiSize = 1;

    // Load nuclide configuration
    NuclideConfig* nucConfig = NuclideConfig::getInstance();
    std::string nuclide_config_file = "./input/nuclides_config_1.txt";

    if (!nucConfig->loadFromFile(nuclide_config_file)) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Failed to load nuclide configuration" << std::endl;
        return 1;
    }

    // Update global nuclide count
    g_num_nuclides = nucConfig->getNumNuclides();

    LDM ldm;

    // Enable grid receptor mode
    ldm.is_grid_receptor_mode = true;
    ldm.grid_count = grid_count;
    ldm.grid_spacing = grid_spacing;

    ldm.loadSimulationConfiguration();

    // Initialize CRAM system with A60.csv matrix (after configuration is loaded)
    std::cout << "[DEBUG] Initializing CRAM system..." << std::endl;
    if (!ldm.initialize_cram_system("./cram/A60.csv")) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "CRAM system initialization failed" << std::endl;
        return 1;
    }
    std::cout << "[DEBUG] CRAM system initialization completed" << std::endl;

    ldm.calculateAverageSettlingVelocity();
    ldm.initializeParticles();

    ldm.loadFlexHeightData();    // Load height data FIRST
    ldm.initializeFlexGFSData(); // Then calculate DRHO using height data

    // Initialize grid receptors
    std::cout << "[GRID] Initializing " << total_receptors << " grid receptors..." << std::endl;
    ldm.initializeGridReceptors(grid_count, grid_spacing);

    ldm.allocateGPUMemory();

    std::cout << "[INFO] Starting simulation..." << std::endl;
    ldm.startTimer();
    ldm.runSimulation();
    ldm.stopTimer();

    // Save grid receptor data
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

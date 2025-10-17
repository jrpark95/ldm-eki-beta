#include "core/ldm.cuh"
#include "physics/ldm_nuclides.cuh"
//#include "ldm_cram.cuh"
//#include "cram_runtime.h"


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

    mpiRank = 1;
    mpiSize = 1;


    // Load nuclide configuration (daughter stress test)
    NuclideConfig* nucConfig = NuclideConfig::getInstance();
    std::string nuclide_config_file = "./input/nuclides_config_1.txt";
    
    if (!nucConfig->loadFromFile(nuclide_config_file)) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Failed to load nuclide configuration" << std::endl;
        return 1;
    }
    
    // Update global nuclide count
    g_num_nuclides = nucConfig->getNumNuclides();
    
    LDM ldm;

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
    ldm.allocateGPUMemory();

    ldm.startTimer();
    ldm.runSimulation();
    ldm.stopTimer();

    // MPI_Finalize();
    return 0;
}
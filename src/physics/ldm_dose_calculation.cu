/******************************************************************************
 * @file ldm_dose_calculation.cu
 * @brief Implementation of Radiation Dose Calculation Module
 *
 * Implements dose calculation formulas based on ADAMO-DOSE methodology:
 *
 * Cloudshine: D_C = C × DCF_cloud × 3600 × dt × SF
 *   - C: Air concentration [Bq/m³]
 *   - DCF_cloud: Cloud shine dose coefficient [Sv·m³/(Bq·s)]
 *   - 3600: Convert hours to seconds
 *   - dt: Time step [hours]
 *   - SF: Shielding factor
 *
 * Groundshine: D_G = Dep × DCF_ground × 3600 × dt × SF
 *   - Dep: Surface deposition [Bq/m²]
 *   - DCF_ground: Ground shine dose coefficient [Sv·m²/(Bq·s)]
 *
 * Inhalation: D_I = C × DCF_inh × BR × 1000 × dt × SF
 *   - DCF_inh: Inhalation dose coefficient [Sv/Bq]
 *   - BR: Breathing rate [m³/hr]
 *   - 1000: Unit conversion factor
 *
 * Total: D_T = D_C + D_G + D_I
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/

#include "ldm_dose_calculation.cuh"
#include <fstream>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <sys/stat.h>
#include <cmath>

// =============================================================================
// DoseResult Implementation
// =============================================================================

void DoseResult::initialize(int nx, int ny, float start_lon, float start_lat, float lon_step, float lat_step) {
    // Initialize effective dose grids
    effdose_C = DoseGrid(nx, ny, start_lon, start_lat, lon_step, lat_step);
    effdose_G = DoseGrid(nx, ny, start_lon, start_lat, lon_step, lat_step);
    for (int i = 0; i < 6; i++) {
        effdose_I[i] = DoseGrid(nx, ny, start_lon, start_lat, lon_step, lat_step);
        effdose_T[i] = DoseGrid(nx, ny, start_lon, start_lat, lon_step, lat_step);
    }

    // Initialize thyroid dose grids
    thydose_C = DoseGrid(nx, ny, start_lon, start_lat, lon_step, lat_step);
    thydose_G = DoseGrid(nx, ny, start_lon, start_lat, lon_step, lat_step);
    for (int i = 0; i < 6; i++) {
        thydose_I[i] = DoseGrid(nx, ny, start_lon, start_lat, lon_step, lat_step);
        thydose_T[i] = DoseGrid(nx, ny, start_lon, start_lat, lon_step, lat_step);
    }

    // Initialize integrated dose grids
    ieffdose_C = DoseGrid(nx, ny, start_lon, start_lat, lon_step, lat_step);
    ieffdose_G = DoseGrid(nx, ny, start_lon, start_lat, lon_step, lat_step);
    for (int i = 0; i < 6; i++) {
        ieffdose_I[i] = DoseGrid(nx, ny, start_lon, start_lat, lon_step, lat_step);
        ieffdose_T[i] = DoseGrid(nx, ny, start_lon, start_lat, lon_step, lat_step);
    }

    ithydose_C = DoseGrid(nx, ny, start_lon, start_lat, lon_step, lat_step);
    ithydose_G = DoseGrid(nx, ny, start_lon, start_lat, lon_step, lat_step);
    for (int i = 0; i < 6; i++) {
        ithydose_I[i] = DoseGrid(nx, ny, start_lon, start_lat, lon_step, lat_step);
        ithydose_T[i] = DoseGrid(nx, ny, start_lon, start_lat, lon_step, lat_step);
    }
}

void DoseResult::resetHourly() {
    effdose_C.reset();
    effdose_G.reset();
    thydose_C.reset();
    thydose_G.reset();
    for (int i = 0; i < 6; i++) {
        effdose_I[i].reset();
        effdose_T[i].reset();
        thydose_I[i].reset();
        thydose_T[i].reset();
    }
}

// =============================================================================
// DoseCalculator Implementation
// =============================================================================

DoseCalculator::DoseCalculator()
    : dcf(nullptr), nuclideIndex(-1), shieldingCondition(0),
      mesh_nx(0), mesh_ny(0), start_lon(0), start_lat(0),
      lon_step(0), lat_step(0), dt_hours(1.0f) {
}

DoseCalculator::~DoseCalculator() {
    // dcf is a singleton, don't delete
}

bool DoseCalculator::initialize(const std::string& nuclide,
                                 int nx, int ny,
                                 float _start_lon, float _start_lat,
                                 float _lon_step, float _lat_step,
                                 float _dt_hours) {
    // Get dose coefficient singleton
    dcf = DoseCoefficients::getInstance();
    if (!dcf) {
        std::cerr << "[ERROR] DoseCoefficients not initialized" << std::endl;
        return false;
    }

    // Find nuclide index
    nuclideIndex = dcf->getNuclideIndex(nuclide);
    if (nuclideIndex < 0) {
        std::cerr << "[ERROR] Nuclide " << nuclide << " not found in dose coefficients" << std::endl;
        return false;
    }

    nuclideName = nuclide;
    mesh_nx = nx;
    mesh_ny = ny;
    start_lon = _start_lon;
    start_lat = _start_lat;
    lon_step = _lon_step;
    lat_step = _lat_step;
    dt_hours = _dt_hours;

    std::cout << "[INFO] DoseCalculator initialized for " << nuclide
              << " (index=" << nuclideIndex << ")" << std::endl;
    std::cout << "       Grid: " << nx << "x" << ny
              << ", lon=[" << start_lon << ":" << lon_step << ":" << (start_lon + (nx-1)*lon_step) << "]"
              << ", lat=[" << start_lat << ":" << lat_step << ":" << (start_lat + (ny-1)*lat_step) << "]"
              << std::endl;

    return true;
}

void DoseCalculator::calculateCloudshine(const float* concentration, DoseResult& result) {
    if (!dcf || nuclideIndex < 0) return;

    const auto& cloudDCF = dcf->getCloudDCF();
    if (nuclideIndex >= static_cast<int>(cloudDCF.size())) return;

    const auto& sf = dcf->getShieldingFactors();
    float shielding = sf.cloudshine[shieldingCondition];

    // DCF values: [Sv·m³/(Bq·s)]
    // Multiply by 3600 to get [Sv·m³/(Bq·hr)]
    float dcf_eff = cloudDCF[nuclideIndex].eff_dose;
    float dcf_thy = cloudDCF[nuclideIndex].thyroid_dose;

    // dose = conc [Bq/m³] × DCF [Sv·m³/(Bq·s)] × 3600 [s/hr] × dt [hr] × shielding
    float factor_eff = dcf_eff * 3600.0f * dt_hours * shielding;
    float factor_thy = dcf_thy * 3600.0f * dt_hours * shielding;

    for (int i = 0; i < mesh_ny; i++) {
        for (int j = 0; j < mesh_nx; j++) {
            int idx = i * mesh_nx + j;
            float conc = concentration[idx];

            if (conc > 0) {
                result.effdose_C.at(i, j) = conc * factor_eff;
                result.thydose_C.at(i, j) = conc * factor_thy;
            }
        }
    }
}

void DoseCalculator::calculateGroundshine(const float* dryDeposition,
                                           const float* wetDeposition,
                                           DoseResult& result) {
    if (!dcf || nuclideIndex < 0) return;

    const auto& groundDCF = dcf->getGroundDCF();
    if (nuclideIndex >= static_cast<int>(groundDCF.size())) return;

    const auto& sf = dcf->getShieldingFactors();
    float shielding = sf.groundshine[shieldingCondition];

    // DCF values for parent and daughter nuclides
    float dcf_eff_p = groundDCF[nuclideIndex].eff_dose;
    float dcf_thy_p = groundDCF[nuclideIndex].thyroid_dose;
    float dcf_eff_d = groundDCF[nuclideIndex].d_eff_dose;
    float dcf_thy_d = groundDCF[nuclideIndex].d_thyroid_dose;

    // dose = dep [Bq/m²] × DCF [Sv·m²/(Bq·s)] × 3600 [s/hr] × dt [hr] × shielding
    float factor_eff_p = dcf_eff_p * 3600.0f * dt_hours * shielding;
    float factor_thy_p = dcf_thy_p * 3600.0f * dt_hours * shielding;
    float factor_eff_d = dcf_eff_d * 3600.0f * dt_hours * shielding;
    float factor_thy_d = dcf_thy_d * 3600.0f * dt_hours * shielding;

    for (int i = 0; i < mesh_ny; i++) {
        for (int j = 0; j < mesh_nx; j++) {
            int idx = i * mesh_nx + j;
            float dep = dryDeposition[idx] + wetDeposition[idx];

            if (dep > 0) {
                // Parent nuclide contribution
                result.effdose_G.at(i, j) = dep * factor_eff_p;
                result.thydose_G.at(i, j) = dep * factor_thy_p;

                // Add daughter nuclide contribution (simplified - no decay chain tracking)
                result.effdose_G.at(i, j) += dep * factor_eff_d;
                result.thydose_G.at(i, j) += dep * factor_thy_d;
            }
        }
    }
}

void DoseCalculator::calculateInhalation(const float* concentration, DoseResult& result) {
    if (!dcf || nuclideIndex < 0) return;

    const auto& sf = dcf->getShieldingFactors();
    float shielding = sf.inhalation[shieldingCondition];

    const auto& br = dcf->getBreathingRate();

    // Calculate for each age group
    for (int age = 0; age < 6; age++) {
        const auto& inhDCF = dcf->getInhalationDCF(age);
        if (nuclideIndex >= static_cast<int>(inhDCF.size())) continue;

        float dcf_eff = inhDCF[nuclideIndex].eff_dose;
        float dcf_thy = inhDCF[nuclideIndex].thyroid_dose;
        float breathing_rate = br.get(age);

        // dose = conc [Bq/m³] × DCF [Sv/Bq] × BR [m³/hr] × 1000 × dt [hr] × shielding
        // Note: The 1000 factor is from the original ADAMO code (unit conversion)
        float factor_eff = dcf_eff * breathing_rate * 1000.0f * dt_hours * shielding;
        float factor_thy = dcf_thy * breathing_rate * 1000.0f * dt_hours * shielding;

        for (int i = 0; i < mesh_ny; i++) {
            for (int j = 0; j < mesh_nx; j++) {
                int idx = i * mesh_nx + j;
                float conc = concentration[idx];

                if (conc > 0) {
                    result.effdose_I[age].at(i, j) = conc * factor_eff;
                    result.thydose_I[age].at(i, j) = conc * factor_thy;
                }
            }
        }
    }
}

void DoseCalculator::calculateTotal(DoseResult& result) {
    for (int age = 0; age < 6; age++) {
        for (int i = 0; i < mesh_ny; i++) {
            for (int j = 0; j < mesh_nx; j++) {
                // Total effective dose = Cloudshine + Groundshine + Inhalation
                result.effdose_T[age].at(i, j) =
                    result.effdose_C.at(i, j) +
                    result.effdose_G.at(i, j) +
                    result.effdose_I[age].at(i, j);

                // Total thyroid dose
                result.thydose_T[age].at(i, j) =
                    result.thydose_C.at(i, j) +
                    result.thydose_G.at(i, j) +
                    result.thydose_I[age].at(i, j);
            }
        }
    }
}

void DoseCalculator::accumulateDoses(DoseResult& result) {
    // Accumulate to integrated grids
    for (int i = 0; i < mesh_ny; i++) {
        for (int j = 0; j < mesh_nx; j++) {
            result.ieffdose_C.at(i, j) += result.effdose_C.at(i, j);
            result.ieffdose_G.at(i, j) += result.effdose_G.at(i, j);
            result.ithydose_C.at(i, j) += result.thydose_C.at(i, j);
            result.ithydose_G.at(i, j) += result.thydose_G.at(i, j);

            for (int age = 0; age < 6; age++) {
                result.ieffdose_I[age].at(i, j) += result.effdose_I[age].at(i, j);
                result.ieffdose_T[age].at(i, j) += result.effdose_T[age].at(i, j);
                result.ithydose_I[age].at(i, j) += result.thydose_I[age].at(i, j);
                result.ithydose_T[age].at(i, j) += result.thydose_T[age].at(i, j);
            }
        }
    }
}

// =============================================================================
// VTK Output Functions
// =============================================================================

// Helper function to swap byte order for VTK binary output
static void swapBytes(float& value) {
    char* bytes = reinterpret_cast<char*>(&value);
    std::swap(bytes[0], bytes[3]);
    std::swap(bytes[1], bytes[2]);
}

void DoseCalculator::outputDoseVTK(const DoseGrid& grid,
                                    const std::string& filename,
                                    const std::string& description) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "[ERROR] Cannot open file for writing: " << filename << std::endl;
        return;
    }

    // VTK header
    outFile << "# vtk DataFile Version 4.2\n";
    outFile << description << "\n";
    outFile << "BINARY\n";
    outFile << "DATASET RECTILINEAR_GRID\n";
    outFile << "DIMENSIONS " << grid.nx << " " << grid.ny << " 1\n";

    // X coordinates (longitude)
    outFile << "X_COORDINATES " << grid.nx << " float\n";
    for (int j = 0; j < grid.nx; j++) {
        float lon = grid.start_lon + j * grid.lon_step;
        swapBytes(lon);
        outFile.write(reinterpret_cast<char*>(&lon), sizeof(float));
    }

    // Y coordinates (latitude)
    outFile << "\nY_COORDINATES " << grid.ny << " float\n";
    for (int i = 0; i < grid.ny; i++) {
        float lat = grid.start_lat + i * grid.lat_step;
        swapBytes(lat);
        outFile.write(reinterpret_cast<char*>(&lat), sizeof(float));
    }

    // Z coordinates (single value for 2D)
    outFile << "\nZ_COORDINATES 1 float\n";
    float zVal = 0.0f;
    swapBytes(zVal);
    outFile.write(reinterpret_cast<char*>(&zVal), sizeof(float));
    outFile << "\n";

    // Point data
    int nPoints = grid.nx * grid.ny;
    outFile << "POINT_DATA " << nPoints << "\n";
    outFile << "SCALARS dose float 1\n";
    outFile << "LOOKUP_TABLE default\n";

    for (int i = 0; i < grid.ny; i++) {
        for (int j = 0; j < grid.nx; j++) {
            float val = grid.at(i, j);
            swapBytes(val);
            outFile.write(reinterpret_cast<char*>(&val), sizeof(float));
        }
    }

    outFile.close();
}

void DoseCalculator::outputAllDosesVTK(const DoseResult& result,
                                        const std::string& outputDir,
                                        int timestep) {
    // Create output directory
    mkdir(outputDir.c_str(), 0777);

    std::ostringstream ss;

    // Output cloudshine dose
    ss.str(""); ss.clear();
    ss << outputDir << "/effdose_C_" << std::setfill('0') << std::setw(5) << timestep << ".vtk";
    outputDoseVTK(result.effdose_C, ss.str(), "Cloudshine Effective Dose [Sv]");

    // Output groundshine dose
    ss.str(""); ss.clear();
    ss << outputDir << "/effdose_G_" << std::setfill('0') << std::setw(5) << timestep << ".vtk";
    outputDoseVTK(result.effdose_G, ss.str(), "Groundshine Effective Dose [Sv]");

    // Output inhalation dose (Adult only for now)
    ss.str(""); ss.clear();
    ss << outputDir << "/effdose_I_adult_" << std::setfill('0') << std::setw(5) << timestep << ".vtk";
    outputDoseVTK(result.effdose_I[0], ss.str(), "Inhalation Effective Dose Adult [Sv]");

    // Output total dose (Adult only for now)
    ss.str(""); ss.clear();
    ss << outputDir << "/effdose_T_adult_" << std::setfill('0') << std::setw(5) << timestep << ".vtk";
    outputDoseVTK(result.effdose_T[0], ss.str(), "Total Effective Dose Adult [Sv]");

    // Output thyroid doses
    ss.str(""); ss.clear();
    ss << outputDir << "/thydose_C_" << std::setfill('0') << std::setw(5) << timestep << ".vtk";
    outputDoseVTK(result.thydose_C, ss.str(), "Cloudshine Thyroid Dose [Sv]");

    ss.str(""); ss.clear();
    ss << outputDir << "/thydose_T_adult_" << std::setfill('0') << std::setw(5) << timestep << ".vtk";
    outputDoseVTK(result.thydose_T[0], ss.str(), "Total Thyroid Dose Adult [Sv]");

    // Output integrated doses
    ss.str(""); ss.clear();
    ss << outputDir << "/ieffdose_T_adult_" << std::setfill('0') << std::setw(5) << timestep << ".vtk";
    outputDoseVTK(result.ieffdose_T[0], ss.str(), "Integrated Total Effective Dose Adult [Sv]");

    ss.str(""); ss.clear();
    ss << outputDir << "/ithydose_T_adult_" << std::setfill('0') << std::setw(5) << timestep << ".vtk";
    outputDoseVTK(result.ithydose_T[0], ss.str(), "Integrated Total Thyroid Dose Adult [Sv]");

    // Print statistics
    printStatistics(result, timestep);
}

void DoseCalculator::printStatistics(const DoseResult& result, int timestep) {
    // Find max values for each dose type
    float maxC = 0, maxG = 0, maxI = 0, maxT = 0;
    float maxThyC = 0, maxThyT = 0;
    float maxIntT = 0, maxIntThyT = 0;

    for (int i = 0; i < mesh_ny; i++) {
        for (int j = 0; j < mesh_nx; j++) {
            maxC = std::max(maxC, result.effdose_C.at(i, j));
            maxG = std::max(maxG, result.effdose_G.at(i, j));
            maxI = std::max(maxI, result.effdose_I[0].at(i, j));
            maxT = std::max(maxT, result.effdose_T[0].at(i, j));
            maxThyC = std::max(maxThyC, result.thydose_C.at(i, j));
            maxThyT = std::max(maxThyT, result.thydose_T[0].at(i, j));
            maxIntT = std::max(maxIntT, result.ieffdose_T[0].at(i, j));
            maxIntThyT = std::max(maxIntThyT, result.ithydose_T[0].at(i, j));
        }
    }

    // Print statistics table

    if(timestep > 215){
        std::cout << "\n[DOSE STATISTICS] Timestep " << timestep << " (" << nuclideName << ")" << std::endl;
        std::cout << "┌─────────────────────┬──────────────────┐" << std::endl;
        std::cout << "│ Dose Type           │ Max Value [Sv]   │" << std::endl;
        std::cout << "├─────────────────────┼──────────────────┤" << std::endl;
        std::cout << "│ Cloudshine (C)      │ " << std::scientific << std::setprecision(3) << std::setw(14) << maxC << " │" << std::endl;
        std::cout << "│ Groundshine (G)     │ " << std::setw(14) << maxG << " │" << std::endl;
        std::cout << "│ Inhalation (I)      │ " << std::setw(14) << maxI << " │" << std::endl;
        std::cout << "│ Total Eff (T)       │ " << std::setw(14) << maxT << " │" << std::endl;
        std::cout << "├─────────────────────┼──────────────────┤" << std::endl;
        std::cout << "│ Thyroid Cloud       │ " << std::setw(14) << maxThyC << " │" << std::endl;
        std::cout << "│ Thyroid Total       │ " << std::setw(14) << maxThyT << " │" << std::endl;
        std::cout << "├─────────────────────┼──────────────────┤" << std::endl;
        std::cout << "│ Integrated Eff      │ " << std::setw(14) << maxIntT << " │" << std::endl;
        std::cout << "│ Integrated Thyroid  │ " << std::setw(14) << maxIntThyT << " │" << std::endl;
        std::cout << "└─────────────────────┴──────────────────┘" << std::endl;
    }

}

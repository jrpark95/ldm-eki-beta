/******************************************************************************
 * @file ldm_dose_calculation.cuh
 * @brief Radiation Dose Calculation Module for LDM
 *
 * Calculates radiation doses from atmospheric dispersion results:
 * - Cloudshine (external dose from air immersion)
 * - Groundshine (external dose from deposited material)
 * - Inhalation (internal dose from breathing contaminated air)
 * - Total dose (sum of all pathways)
 *
 * Based on ADAMO-DOSE_FLEXPART methodology (FNC Technology, 2016-2018)
 *
 * Units:
 * - Input concentration: Bq/m³
 * - Input deposition: Bq/m²
 * - Output dose: Sv (Sievert)
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/

#ifndef LDM_DOSE_CALCULATION_CUH
#define LDM_DOSE_CALCULATION_CUH

#include "ldm_dose_coefficients.cuh"
#include <vector>
#include <string>

/**
 * @struct DoseGrid
 * @brief 2D grid storing dose values for visualization
 */
struct DoseGrid {
    int nx;                     // Number of grid points in longitude
    int ny;                     // Number of grid points in latitude
    float start_lon;            // Starting longitude [degrees]
    float start_lat;            // Starting latitude [degrees]
    float lon_step;             // Longitude resolution [degrees]
    float lat_step;             // Latitude resolution [degrees]
    std::vector<float> data;    // Dose values [ny × nx]

    DoseGrid() : nx(0), ny(0), start_lon(0), start_lat(0), lon_step(0), lat_step(0) {}

    DoseGrid(int _nx, int _ny, float _start_lon, float _start_lat, float _lon_step, float _lat_step)
        : nx(_nx), ny(_ny), start_lon(_start_lon), start_lat(_start_lat),
          lon_step(_lon_step), lat_step(_lat_step) {
        data.resize(nx * ny, 0.0f);
    }

    void reset() {
        std::fill(data.begin(), data.end(), 0.0f);
    }

    float& at(int i, int j) {
        return data[i * nx + j];
    }

    const float& at(int i, int j) const {
        return data[i * nx + j];
    }
};

/**
 * @struct DoseResult
 * @brief Complete dose calculation results for all pathways
 */
struct DoseResult {
    // Effective dose grids
    DoseGrid effdose_C;     // Cloudshine effective dose [Sv]
    DoseGrid effdose_G;     // Groundshine effective dose [Sv]
    DoseGrid effdose_I[6];  // Inhalation effective dose by age group [Sv]
    DoseGrid effdose_T[6];  // Total effective dose by age group [Sv]

    // Thyroid dose grids
    DoseGrid thydose_C;     // Cloudshine thyroid dose [Sv]
    DoseGrid thydose_G;     // Groundshine thyroid dose [Sv]
    DoseGrid thydose_I[6];  // Inhalation thyroid dose by age group [Sv]
    DoseGrid thydose_T[6];  // Total thyroid dose by age group [Sv]

    // Integrated (cumulative) doses
    DoseGrid ieffdose_C;    // Integrated cloudshine effective dose
    DoseGrid ieffdose_G;    // Integrated groundshine effective dose
    DoseGrid ieffdose_I[6]; // Integrated inhalation effective dose
    DoseGrid ieffdose_T[6]; // Integrated total effective dose

    DoseGrid ithydose_C;    // Integrated cloudshine thyroid dose
    DoseGrid ithydose_G;    // Integrated groundshine thyroid dose
    DoseGrid ithydose_I[6]; // Integrated inhalation thyroid dose
    DoseGrid ithydose_T[6]; // Integrated total thyroid dose

    void initialize(int nx, int ny, float start_lon, float start_lat, float lon_step, float lat_step);
    void resetHourly();  // Reset hourly (non-integrated) grids
};

/**
 * @class DoseCalculator
 * @brief Main class for radiation dose calculations
 *
 * Calculates doses from concentration and deposition grids using
 * dose conversion factors loaded from input/model1/*.dat files.
 */
class DoseCalculator {
private:
    DoseCoefficients* dcf;          // Dose conversion factors
    int nuclideIndex;               // Index of current nuclide in DCF tables
    std::string nuclideName;        // Current nuclide name (e.g., "Cs-137")
    int shieldingCondition;         // 0=Normal, 1=Sheltering, 2=Evacuation

    // Grid parameters
    int mesh_nx, mesh_ny;
    float start_lon, start_lat;
    float lon_step, lat_step;

    // Time step for integration (hours)
    float dt_hours;

public:
    DoseCalculator();
    ~DoseCalculator();

    /**
     * @brief Initialize dose calculator
     * @param nuclide Nuclide name (e.g., "Cs-137")
     * @param nx Grid points in longitude
     * @param ny Grid points in latitude
     * @param _start_lon Starting longitude
     * @param _start_lat Starting latitude
     * @param _lon_step Longitude resolution
     * @param _lat_step Latitude resolution
     * @param _dt_hours Time step in hours
     * @return true if successful
     */
    bool initialize(const std::string& nuclide,
                    int nx, int ny,
                    float _start_lon, float _start_lat,
                    float _lon_step, float _lat_step,
                    float _dt_hours = 1.0f);

    /**
     * @brief Set shielding condition
     * @param condition 0=Normal (outdoor), 1=Sheltering (indoor), 2=Evacuation
     */
    void setShieldingCondition(int condition) { shieldingCondition = condition; }

    /**
     * @brief Calculate cloudshine dose from air concentration
     *
     * Formula: dose = conc × DCF_cloud × 3600 × dt_hours × shielding_factor
     *
     * @param concentration Air concentration grid [Bq/m³]
     * @param result Output dose result structure
     */
    void calculateCloudshine(const float* concentration, DoseResult& result);

    /**
     * @brief Calculate groundshine dose from deposition
     *
     * Formula: dose = dep × DCF_ground × 3600 × dt_hours × shielding_factor
     *
     * @param dryDeposition Dry deposition grid [Bq/m²]
     * @param wetDeposition Wet deposition grid [Bq/m²]
     * @param result Output dose result structure
     */
    void calculateGroundshine(const float* dryDeposition,
                              const float* wetDeposition,
                              DoseResult& result);

    /**
     * @brief Calculate inhalation dose from air concentration
     *
     * Formula: dose = conc × DCF_inhalation × breathing_rate × dt_hours × shielding_factor
     *
     * @param concentration Air concentration grid [Bq/m³]
     * @param result Output dose result structure
     */
    void calculateInhalation(const float* concentration, DoseResult& result);

    /**
     * @brief Calculate total dose (sum of all pathways)
     * @param result Dose result with C, G, I already calculated
     */
    void calculateTotal(DoseResult& result);

    /**
     * @brief Accumulate hourly doses to integrated doses
     * @param result Dose result structure
     */
    void accumulateDoses(DoseResult& result);

    /**
     * @brief Output dose grid to VTK file
     * @param grid Dose grid to output
     * @param filename Output filename
     * @param description Data description for VTK header
     */
    void outputDoseVTK(const DoseGrid& grid,
                       const std::string& filename,
                       const std::string& description);

    /**
     * @brief Output all dose results to VTK files
     * @param result Dose result structure
     * @param outputDir Output directory
     * @param timestep Current timestep for filename
     */
    void outputAllDosesVTK(const DoseResult& result,
                           const std::string& outputDir,
                           int timestep);

    /**
     * @brief Print dose statistics (max values) to console
     * @param result Dose result structure
     * @param timestep Current timestep
     */
    void printStatistics(const DoseResult& result, int timestep);

    // Getters
    int getNuclideIndex() const { return nuclideIndex; }
    const std::string& getNuclideName() const { return nuclideName; }
};

#endif // LDM_DOSE_CALCULATION_CUH

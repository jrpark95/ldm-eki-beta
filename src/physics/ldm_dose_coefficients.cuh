/******************************************************************************
 * @file ldm_dose_coefficients.cuh
 * @brief Dose Conversion Factor (DCF) data structures and loader
 *
 * This module manages dose conversion factors for radiation dose calculation:
 * - Cloud shine (external dose from air immersion)
 * - Ground shine (external dose from deposited material)
 * - Inhalation (internal dose from breathing contaminated air)
 *
 * Data is loaded from input/model1/*.dat files.
 *
 * Reference: ADAMO-DOSE_FLEXPART (FNC Technology, 2016-2018)
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/

#ifndef LDM_DOSE_COEFFICIENTS_CUH
#define LDM_DOSE_COEFFICIENTS_CUH

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

/// Number of nuclides in dose coefficient database
#define NUM_DCF_NUCLIDES 31

/// Number of age groups (Adult, 15y, 10y, 5y, 1y, 3m)
#define NUM_AGE_GROUPS 6

/**
 * @struct BreathingRate
 * @brief Breathing rates for different age groups [m³/hr]
 */
struct BreathingRate {
    float adult;    // A: 0.917 m³/hr
    float y15;      // 15y: 0.833 m³/hr
    float y10;      // 10y: 0.625 m³/hr
    float y5;       // 5y: 0.363 m³/hr
    float y1;       // 1y: 0.217 m³/hr
    float m3;       // 3m: 0.121 m³/hr

    float get(int age_index) const {
        switch(age_index) {
            case 0: return adult;
            case 1: return y15;
            case 2: return y10;
            case 3: return y5;
            case 4: return y1;
            case 5: return m3;
            default: return adult;
        }
    }
};

/**
 * @struct DecayConstantData
 * @brief Decay constants for parent and daughter nuclides
 */
struct DecayConstantData {
    char name[16];          // Nuclide name (e.g., "I-131")
    int z;                  // Atomic number
    float constant;         // Decay constant [/hr] for parent
    float d_constant;       // Decay constant [/hr] for daughter
    float branch_ratio;     // Branching ratio
};

/**
 * @struct CloudDCF
 * @brief Cloud shine dose conversion factors (external dose from air immersion)
 *
 * Units: Sv·m³/(Bq·s) - converts activity concentration to dose rate
 */
struct CloudDCF {
    char name[16];          // Nuclide name
    int z;                  // Atomic number
    int mass;               // Mass number
    float eff_dose;         // Effective dose coefficient [Sv·m³/(Bq·s)]
    float thyroid_dose;     // Thyroid dose coefficient [Sv·m³/(Bq·s)]
};

/**
 * @struct GroundDCF
 * @brief Ground shine dose conversion factors (external dose from deposition)
 *
 * Units: Sv·m²/(Bq·s) - converts surface activity to dose rate
 */
struct GroundDCF {
    char name[16];          // Nuclide name
    int z;                  // Atomic number
    int mass;               // Mass number
    float eff_dose;         // Effective dose coefficient (parent) [Sv·m²/(Bq·s)]
    float thyroid_dose;     // Thyroid dose coefficient (parent) [Sv·m²/(Bq·s)]
    float d_eff_dose;       // Effective dose coefficient (daughter) [Sv·m²/(Bq·s)]
    float d_thyroid_dose;   // Thyroid dose coefficient (daughter) [Sv·m²/(Bq·s)]
};

/**
 * @struct InhalationDCF
 * @brief Inhalation dose conversion factors for one age group
 *
 * Units: Sv/Bq - converts inhaled activity to committed dose
 */
struct InhalationDCF {
    char name[16];          // Nuclide name
    int z;                  // Atomic number
    int mass;               // Mass number
    float eff_dose;         // Effective dose coefficient [Sv/Bq]
    float thyroid_dose;     // Thyroid dose coefficient [Sv/Bq]
};

/**
 * @struct ShieldingFactors
 * @brief Shielding factors for different exposure pathways and conditions
 *
 * Reduces dose based on shelter type:
 * - Normal: Outdoor, no protection
 * - Sheltering: Indoor (building)
 * - Evacuation: Moving
 */
struct ShieldingFactors {
    float cloudshine[3];    // [Normal, Sheltering, Evacuation]
    float inhalation[3];    // [Normal, Sheltering, Evacuation]
    float skin[3];          // [Normal, Sheltering, Evacuation]
    float groundshine[3];   // [Normal, Sheltering, Evacuation]

    // IERNet scaling factors
    float iernet_cloud;     // Cloudshine factor for IERNet
    float iernet_ground;    // Groundshine factor for IERNet
};

/**
 * @class DoseCoefficients
 * @brief Singleton class to manage all dose conversion factors
 *
 * Loads and provides access to:
 * - Breathing rates by age
 * - Decay constants (parent and daughter)
 * - Cloud shine DCFs
 * - Ground shine DCFs
 * - Inhalation DCFs (6 age groups)
 * - Shielding factors
 */
class DoseCoefficients {
private:
    static DoseCoefficients* instance;

    // Data storage
    BreathingRate breathingRate;
    std::vector<DecayConstantData> decayConstants;
    std::vector<CloudDCF> cloudDCF;
    std::vector<GroundDCF> groundDCF;
    std::vector<InhalationDCF> inhalationDCF_A;     // Adult
    std::vector<InhalationDCF> inhalationDCF_15y;   // 15 years
    std::vector<InhalationDCF> inhalationDCF_10y;   // 10 years
    std::vector<InhalationDCF> inhalationDCF_5y;    // 5 years
    std::vector<InhalationDCF> inhalationDCF_1y;    // 1 year
    std::vector<InhalationDCF> inhalationDCF_3m;    // 3 months
    ShieldingFactors shieldingFactors;

    // Nuclide name to index mapping
    std::map<std::string, int> nuclideIndex;

    // Private constructor (singleton)
    DoseCoefficients();

    // Internal parsing helpers
    bool parseBreathingRate(const std::string& filepath);
    bool parseDecayConstants(const std::string& filepath);
    bool parseCloudDCF(const std::string& filepath);
    bool parseGroundDCF(const std::string& filepath);
    bool parseInhalationDCF(const std::string& filepath, std::vector<InhalationDCF>& dcf);
    bool parseShieldingFactors(const std::string& filepath);

    // Helper to parse scientific notation
    float parseScientific(const std::string& str);

public:
    ~DoseCoefficients();

    /**
     * @brief Get singleton instance
     */
    static DoseCoefficients* getInstance();

    /**
     * @brief Load all dose coefficient data from input directory
     * @param basePath Base path to data files (e.g., "./input/model1/")
     * @return true if all files loaded successfully
     */
    bool loadAllData(const std::string& basePath);

    // Accessors
    const BreathingRate& getBreathingRate() const { return breathingRate; }
    const std::vector<DecayConstantData>& getDecayConstants() const { return decayConstants; }
    const std::vector<CloudDCF>& getCloudDCF() const { return cloudDCF; }
    const std::vector<GroundDCF>& getGroundDCF() const { return groundDCF; }
    const ShieldingFactors& getShieldingFactors() const { return shieldingFactors; }

    /**
     * @brief Get inhalation DCF for specific age group
     * @param ageIndex 0=Adult, 1=15y, 2=10y, 3=5y, 4=1y, 5=3m
     */
    const std::vector<InhalationDCF>& getInhalationDCF(int ageIndex) const;

    /**
     * @brief Get nuclide index by name
     * @return Index (0-30) or -1 if not found
     */
    int getNuclideIndex(const std::string& name) const;

    /**
     * @brief Get number of loaded nuclides
     */
    int getNumNuclides() const { return static_cast<int>(decayConstants.size()); }

    /**
     * @brief Print loaded data summary for verification
     */
    void printSummary() const;
};

#endif // LDM_DOSE_COEFFICIENTS_CUH

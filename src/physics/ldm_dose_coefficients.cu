/******************************************************************************
 * @file ldm_dose_coefficients.cu
 * @brief Implementation of Dose Conversion Factor loader and manager
 *
 * Parses dose coefficient data files from input/model1/:
 * - Breathing_Rate.dat
 * - Decay_constant.dat
 * - ExDCF_Cloud.dat
 * - ExDCF_Ground.dat
 * - InDCF_Inhalation_*.dat (6 age groups)
 * - Shielding_Factors.dat
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/

#include "ldm_dose_coefficients.cuh"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <iomanip>

// Singleton instance
DoseCoefficients* DoseCoefficients::instance = nullptr;

// =============================================================================
// Constructor / Destructor
// =============================================================================

DoseCoefficients::DoseCoefficients() {
    // Initialize breathing rate to zeros
    breathingRate = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Initialize shielding factors to default (no shielding)
    for (int i = 0; i < 3; i++) {
        shieldingFactors.cloudshine[i] = 1.0f;
        shieldingFactors.inhalation[i] = 1.0f;
        shieldingFactors.skin[i] = 1.0f;
        shieldingFactors.groundshine[i] = 1.0f;
    }
    shieldingFactors.iernet_cloud = 1.0f;
    shieldingFactors.iernet_ground = 1.0f;
}

DoseCoefficients::~DoseCoefficients() {
    // Clean up vectors
    decayConstants.clear();
    cloudDCF.clear();
    groundDCF.clear();
    inhalationDCF_A.clear();
    inhalationDCF_15y.clear();
    inhalationDCF_10y.clear();
    inhalationDCF_5y.clear();
    inhalationDCF_1y.clear();
    inhalationDCF_3m.clear();
    nuclideIndex.clear();
}

DoseCoefficients* DoseCoefficients::getInstance() {
    if (instance == nullptr) {
        instance = new DoseCoefficients();
    }
    return instance;
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * @brief Parse scientific notation string to float
 * @param str String like "1.23E-10" or "1.23e-10"
 */
float DoseCoefficients::parseScientific(const std::string& str) {
    if (str.empty()) return 0.0f;

    // Trim whitespace
    std::string s = str;
    s.erase(0, s.find_first_not_of(" \t\r\n"));
    s.erase(s.find_last_not_of(" \t\r\n") + 1);

    if (s.empty()) return 0.0f;

    try {
        return std::stof(s);
    } catch (...) {
        return 0.0f;
    }
}

/**
 * @brief Trim whitespace from string
 */
static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

/**
 * @brief Split string by tab delimiter
 */
static std::vector<std::string> splitByTab(const std::string& line) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, '\t')) {
        tokens.push_back(trim(token));
    }
    return tokens;
}

// =============================================================================
// Parsing Functions
// =============================================================================

bool DoseCoefficients::parseBreathingRate(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open: " << filepath << std::endl;
        return false;
    }

    std::string line;
    // Skip header line
    std::getline(file, line);

    // Read data lines
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        auto tokens = splitByTab(line);
        if (tokens.size() < 2) continue;

        std::string ageGroup = tokens[0];
        float rate = parseScientific(tokens[1]);

        if (ageGroup == "A") breathingRate.adult = rate;
        else if (ageGroup == "15y") breathingRate.y15 = rate;
        else if (ageGroup == "10y") breathingRate.y10 = rate;
        else if (ageGroup == "5y") breathingRate.y5 = rate;
        else if (ageGroup == "1y") breathingRate.y1 = rate;
        else if (ageGroup == "3m") breathingRate.m3 = rate;
    }

    file.close();
    std::cout << "[INFO] Loaded breathing rates from: " << filepath << std::endl;
    return true;
}

bool DoseCoefficients::parseDecayConstants(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open: " << filepath << std::endl;
        return false;
    }

    std::string line;
    // Skip header line
    std::getline(file, line);

    int idx = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        auto tokens = splitByTab(line);
        if (tokens.size() < 4) continue;

        // Skip empty tokens
        std::vector<std::string> validTokens;
        for (const auto& t : tokens) {
            if (!t.empty()) validTokens.push_back(t);
        }
        if (validTokens.size() < 4) continue;

        DecayConstantData data;
        strncpy(data.name, validTokens[0].c_str(), sizeof(data.name) - 1);
        data.name[sizeof(data.name) - 1] = '\0';

        try {
            data.z = std::stoi(validTokens[1]);
        } catch (...) {
            std::cerr << "[WARNING] Invalid Z value for nuclide: " << validTokens[0] << std::endl;
            continue;
        }

        data.constant = parseScientific(validTokens[2]);
        data.d_constant = parseScientific(validTokens[3]);
        data.branch_ratio = (validTokens.size() > 4) ? parseScientific(validTokens[4]) : 1.0f;

        decayConstants.push_back(data);

        // Build index map
        nuclideIndex[validTokens[0]] = idx++;
    }

    file.close();
    std::cout << "[INFO] Loaded " << decayConstants.size() << " nuclide decay constants from: " << filepath << std::endl;
    return true;
}

bool DoseCoefficients::parseCloudDCF(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open: " << filepath << std::endl;
        return false;
    }

    std::string line;
    // Skip header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        auto tokens = splitByTab(line);

        // Filter empty tokens
        std::vector<std::string> validTokens;
        for (const auto& t : tokens) {
            if (!t.empty()) validTokens.push_back(t);
        }
        if (validTokens.size() < 5) continue;

        CloudDCF data;
        strncpy(data.name, validTokens[0].c_str(), sizeof(data.name) - 1);
        data.name[sizeof(data.name) - 1] = '\0';

        try {
            data.z = std::stoi(validTokens[1]);
            data.mass = std::stoi(validTokens[2]);
        } catch (...) {
            continue;
        }

        data.eff_dose = parseScientific(validTokens[3]);
        data.thyroid_dose = parseScientific(validTokens[4]);

        cloudDCF.push_back(data);
    }

    file.close();
    std::cout << "[INFO] Loaded " << cloudDCF.size() << " cloud DCFs from: " << filepath << std::endl;
    return true;
}

bool DoseCoefficients::parseGroundDCF(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open: " << filepath << std::endl;
        return false;
    }

    std::string line;
    // Skip header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        auto tokens = splitByTab(line);

        // Filter empty tokens
        std::vector<std::string> validTokens;
        for (const auto& t : tokens) {
            if (!t.empty()) validTokens.push_back(t);
        }
        if (validTokens.size() < 7) continue;

        GroundDCF data;
        strncpy(data.name, validTokens[0].c_str(), sizeof(data.name) - 1);
        data.name[sizeof(data.name) - 1] = '\0';

        try {
            data.z = std::stoi(validTokens[1]);
            data.mass = std::stoi(validTokens[2]);
        } catch (...) {
            continue;
        }

        data.eff_dose = parseScientific(validTokens[3]);
        data.thyroid_dose = parseScientific(validTokens[4]);
        data.d_eff_dose = parseScientific(validTokens[5]);
        data.d_thyroid_dose = parseScientific(validTokens[6]);

        groundDCF.push_back(data);
    }

    file.close();
    std::cout << "[INFO] Loaded " << groundDCF.size() << " ground DCFs from: " << filepath << std::endl;
    return true;
}

bool DoseCoefficients::parseInhalationDCF(const std::string& filepath, std::vector<InhalationDCF>& dcf) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open: " << filepath << std::endl;
        return false;
    }

    dcf.clear();

    std::string line;
    // Skip header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        auto tokens = splitByTab(line);

        // Filter empty tokens
        std::vector<std::string> validTokens;
        for (const auto& t : tokens) {
            if (!t.empty()) validTokens.push_back(t);
        }
        if (validTokens.size() < 6) continue;

        InhalationDCF data;
        strncpy(data.name, validTokens[0].c_str(), sizeof(data.name) - 1);
        data.name[sizeof(data.name) - 1] = '\0';

        try {
            data.z = std::stoi(validTokens[1]);
            data.mass = std::stoi(validTokens[2]);
        } catch (...) {
            continue;
        }

        // validTokens[3] is age group (skip)
        data.eff_dose = parseScientific(validTokens[4]);
        data.thyroid_dose = parseScientific(validTokens[5]);

        dcf.push_back(data);
    }

    file.close();
    std::cout << "[INFO] Loaded " << dcf.size() << " inhalation DCFs from: " << filepath << std::endl;
    return true;
}

bool DoseCoefficients::parseShieldingFactors(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open: " << filepath << std::endl;
        return false;
    }

    std::string line;
    // Skip header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        auto tokens = splitByTab(line);

        // Filter empty tokens and get valid values
        std::vector<std::string> validTokens;
        for (const auto& t : tokens) {
            if (!t.empty()) validTokens.push_back(t);
        }
        if (validTokens.empty()) continue;

        std::string name = validTokens[0];

        if (name == "Cloudshine" && validTokens.size() >= 4) {
            shieldingFactors.cloudshine[0] = parseScientific(validTokens[1]);
            shieldingFactors.cloudshine[1] = parseScientific(validTokens[2]);
            shieldingFactors.cloudshine[2] = parseScientific(validTokens[3]);
        }
        else if (name == "Inhalation" && validTokens.size() >= 4) {
            shieldingFactors.inhalation[0] = parseScientific(validTokens[1]);
            shieldingFactors.inhalation[1] = parseScientific(validTokens[2]);
            shieldingFactors.inhalation[2] = parseScientific(validTokens[3]);
        }
        else if (name == "Skin_protection" && validTokens.size() >= 4) {
            shieldingFactors.skin[0] = parseScientific(validTokens[1]);
            shieldingFactors.skin[1] = parseScientific(validTokens[2]);
            shieldingFactors.skin[2] = parseScientific(validTokens[3]);
        }
        else if (name == "Groundshine" && validTokens.size() >= 4) {
            shieldingFactors.groundshine[0] = parseScientific(validTokens[1]);
            shieldingFactors.groundshine[1] = parseScientific(validTokens[2]);
            shieldingFactors.groundshine[2] = parseScientific(validTokens[3]);
        }
        else if (name == "IERNet" && validTokens.size() >= 3) {
            shieldingFactors.iernet_cloud = parseScientific(validTokens[1]);
            shieldingFactors.iernet_ground = parseScientific(validTokens[2]);
        }
    }

    file.close();
    std::cout << "[INFO] Loaded shielding factors from: " << filepath << std::endl;
    return true;
}

// =============================================================================
// Public Interface
// =============================================================================

bool DoseCoefficients::loadAllData(const std::string& basePath) {
    std::string path = basePath;
    if (path.back() != '/') path += '/';

    bool success = true;

    // Load breathing rate
    if (!parseBreathingRate(path + "Breathing_Rate.dat")) {
        std::cerr << "[WARNING] Failed to load Breathing_Rate.dat" << std::endl;
        success = false;
    }

    // Load decay constants
    if (!parseDecayConstants(path + "Decay_constant.dat")) {
        std::cerr << "[WARNING] Failed to load Decay_constant.dat" << std::endl;
        success = false;
    }

    // Load cloud DCF
    if (!parseCloudDCF(path + "ExDCF_Cloud.dat")) {
        std::cerr << "[WARNING] Failed to load ExDCF_Cloud.dat" << std::endl;
        success = false;
    }

    // Load ground DCF
    if (!parseGroundDCF(path + "ExDCF_Ground.dat")) {
        std::cerr << "[WARNING] Failed to load ExDCF_Ground.dat" << std::endl;
        success = false;
    }

    // Load inhalation DCFs for all age groups
    if (!parseInhalationDCF(path + "InDCF_Inhalation_A.dat", inhalationDCF_A)) {
        std::cerr << "[WARNING] Failed to load InDCF_Inhalation_A.dat" << std::endl;
        success = false;
    }
    if (!parseInhalationDCF(path + "InDCF_Inhalation_15y.dat", inhalationDCF_15y)) {
        std::cerr << "[WARNING] Failed to load InDCF_Inhalation_15y.dat" << std::endl;
        success = false;
    }
    if (!parseInhalationDCF(path + "InDCF_Inhalation_10y.dat", inhalationDCF_10y)) {
        std::cerr << "[WARNING] Failed to load InDCF_Inhalation_10y.dat" << std::endl;
        success = false;
    }
    if (!parseInhalationDCF(path + "InDCF_Inhalation_5y.dat", inhalationDCF_5y)) {
        std::cerr << "[WARNING] Failed to load InDCF_Inhalation_5y.dat" << std::endl;
        success = false;
    }
    if (!parseInhalationDCF(path + "InDCF_Inhalation_1y.dat", inhalationDCF_1y)) {
        std::cerr << "[WARNING] Failed to load InDCF_Inhalation_1y.dat" << std::endl;
        success = false;
    }
    if (!parseInhalationDCF(path + "InDCF_Inhalation_3m.dat", inhalationDCF_3m)) {
        std::cerr << "[WARNING] Failed to load InDCF_Inhalation_3m.dat" << std::endl;
        success = false;
    }

    // Load shielding factors
    if (!parseShieldingFactors(path + "Shielding_Factors.dat")) {
        std::cerr << "[WARNING] Failed to load Shielding_Factors.dat" << std::endl;
        success = false;
    }

    return success;
}

const std::vector<InhalationDCF>& DoseCoefficients::getInhalationDCF(int ageIndex) const {
    switch (ageIndex) {
        case 0: return inhalationDCF_A;
        case 1: return inhalationDCF_15y;
        case 2: return inhalationDCF_10y;
        case 3: return inhalationDCF_5y;
        case 4: return inhalationDCF_1y;
        case 5: return inhalationDCF_3m;
        default: return inhalationDCF_A;
    }
}

int DoseCoefficients::getNuclideIndex(const std::string& name) const {
    auto it = nuclideIndex.find(name);
    if (it != nuclideIndex.end()) {
        return it->second;
    }
    return -1;
}

void DoseCoefficients::printSummary() const {
    std::cout << "\n========== Dose Coefficients Summary ==========\n" << std::endl;

    // Breathing rates
    std::cout << "--- Breathing Rates [m³/hr] ---" << std::endl;
    std::cout << "  Adult: " << breathingRate.adult << std::endl;
    std::cout << "  15y:   " << breathingRate.y15 << std::endl;
    std::cout << "  10y:   " << breathingRate.y10 << std::endl;
    std::cout << "  5y:    " << breathingRate.y5 << std::endl;
    std::cout << "  1y:    " << breathingRate.y1 << std::endl;
    std::cout << "  3m:    " << breathingRate.m3 << std::endl;

    // Shielding factors
    std::cout << "\n--- Shielding Factors [Normal/Sheltering/Evacuation] ---" << std::endl;
    std::cout << "  Cloudshine:  " << shieldingFactors.cloudshine[0] << " / "
              << shieldingFactors.cloudshine[1] << " / " << shieldingFactors.cloudshine[2] << std::endl;
    std::cout << "  Inhalation:  " << shieldingFactors.inhalation[0] << " / "
              << shieldingFactors.inhalation[1] << " / " << shieldingFactors.inhalation[2] << std::endl;
    std::cout << "  Groundshine: " << shieldingFactors.groundshine[0] << " / "
              << shieldingFactors.groundshine[1] << " / " << shieldingFactors.groundshine[2] << std::endl;
    std::cout << "  IERNet: cloud=" << shieldingFactors.iernet_cloud
              << ", ground=" << shieldingFactors.iernet_ground << std::endl;

    // Nuclide summary (first 5)
    std::cout << "\n--- Nuclides (first 5 of " << decayConstants.size() << ") ---" << std::endl;
    std::cout << std::setw(10) << "Name" << std::setw(12) << "Lambda(/hr)"
              << std::setw(12) << "d_Lambda" << std::setw(14) << "CloudDCF(Eff)"
              << std::setw(14) << "GroundDCF(Eff)" << std::endl;

    int count = std::min(5, static_cast<int>(decayConstants.size()));
    for (int i = 0; i < count; i++) {
        std::cout << std::setw(10) << decayConstants[i].name
                  << std::setw(12) << std::scientific << std::setprecision(2) << decayConstants[i].constant
                  << std::setw(12) << decayConstants[i].d_constant;

        if (i < static_cast<int>(cloudDCF.size())) {
            std::cout << std::setw(14) << cloudDCF[i].eff_dose;
        }
        if (i < static_cast<int>(groundDCF.size())) {
            std::cout << std::setw(14) << groundDCF[i].eff_dose;
        }
        std::cout << std::endl;
    }

    std::cout << "\n================================================\n" << std::endl;
}

/******************************************************************************
 * @file ldm_config.cuh
 * @brief Configuration file reader for LDM-EKI simulation system
 *
 * @details Provides the ConfigReader class for parsing key-value configuration
 *          files used by the LDM-EKI system. This header-only utility class
 *          enables type-safe configuration file parsing with support for:
 *
 *          **Supported Value Types:**
 *          - String: Raw text values
 *          - Integer: int type (std::stoi)
 *          - Float: float type (std::stof)
 *          - Double: double type (std::stod)
 *          - Arrays: Comma-separated values (string/float/double)
 *
 *          **File Format:**
 *          - Key-value pairs: `KEY: value`
 *          - Comment lines: Lines starting with `#`
 *          - Whitespace: Leading/trailing spaces trimmed automatically
 *          - Encoding: ASCII/UTF-8 compatible
 *
 *          **Error Handling:**
 *          - Missing keys return default values (no exception)
 *          - Parse errors print warnings and use defaults
 *          - File open failures return false from loadConfig()
 *
 * @note This is a header-only file - no corresponding .cu implementation
 * @note Uses std::map for O(log n) key lookup performance
 * @note Thread-safe for read-only operations after loadConfig() completes
 *
 * @example Basic usage
 * @code
 *   ConfigReader config;
 *   if (config.loadConfig("input/setting.txt")) {
 *       float dt = config.getFloat("time_step", 1.0f);
 *       int particles = config.getInt("total_particles", 10000);
 *       std::string model = config.getString("model_name", "default");
 *   }
 * @endcode
 *
 * @example Array usage
 * @code
 *   std::vector<float> emissions = config.getFloatArray("emission_rates");
 *   std::vector<std::string> files = config.getStringArray("input_files");
 * @endcode
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>

/**
 * @brief Configuration file parser with type-safe value retrieval
 *
 * @details ConfigReader parses simple key-value configuration files and
 *          provides type-safe getter methods with default value fallback.
 *          The parser is lenient: whitespace variations and missing keys
 *          are handled gracefully without throwing exceptions.
 *
 * **Configuration File Format:**
 * ```
 * # Comment line (ignored)
 * KEY1: value1
 * KEY2: value2, value3, value4  # Comma-separated array
 * KEY3:  value_with_spaces       # Leading/trailing spaces trimmed
 * ```
 *
 * **Internal Storage:**
 * - Uses std::map<std::string, std::string> for key-value pairs
 * - All values stored as strings, converted on-demand by getters
 * - Lookup complexity: O(log n) where n = number of keys
 *
 * **Thread Safety:**
 * - Safe for concurrent reads after loadConfig() completes
 * - NOT safe if loadConfig() called concurrently
 * - Recommend: load once at startup, then read-only access
 */
class ConfigReader {
private:
    std::map<std::string, std::string> config_map;  ///< Key-value storage - Internal string representation

public:
    /**
     * @brief Load configuration from file
     *
     * Parses a configuration file in KEY: value format. Each line is processed
     * as follows:
     * 1. Empty lines and lines starting with '#' are skipped
     * 2. Lines are split at first ':' character
     * 3. Key (left) and value (right) are trimmed of whitespace
     * 4. Key-value pair stored in internal map
     *
     * Multiple calls to loadConfig() append to existing configuration without
     * clearing previous values. Keys from newer files overwrite older ones.
     *
     * @param[in] filename Path to configuration file
     *                     - Relative or absolute path
     *                     - Must be readable text file
     *                     - ASCII or UTF-8 encoding
     *
     * @return true if file opened successfully, false otherwise
     *         - Returns true even if file is empty or all lines are comments
     *         - Returns false only for file open failures (not found, permissions, etc.)
     *
     * @note Parsing errors for individual lines are silently ignored
     * @note Lines without ':' separator are skipped without warning
     * @note Duplicate keys: later values overwrite earlier ones
     *
     * @warning File must exist and be readable - no automatic fallback
     */
    bool loadConfig(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open config file: " << filename << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') continue;

            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);

                // Trim whitespace and newline characters
                key.erase(0, key.find_first_not_of(" \t\r\n"));
                key.erase(key.find_last_not_of(" \t\r\n") + 1);
                value.erase(0, value.find_first_not_of(" \t\r\n"));
                value.erase(value.find_last_not_of(" \t\r\n") + 1);

                config_map[key] = value;
            }
        }
        file.close();
        return true;
    }

    /**
     * @brief Retrieve string value for given key
     *
     * @param[in] key Configuration key to look up
     * @param[in] defaultValue Value returned if key not found [default: ""]
     *
     * @return String value associated with key, or defaultValue if key missing
     *
     * @note No parsing/conversion performed - returns raw string value
     * @note Empty strings are valid values (distinct from missing key)
     */
    std::string getString(const std::string& key, const std::string& defaultValue = "") {
        auto it = config_map.find(key);
        return (it != config_map.end()) ? it->second : defaultValue;
    }

    /**
     * @brief Retrieve double value for given key
     *
     * Attempts to parse value string as double using std::stod. If parsing
     * fails, prints warning and returns default value.
     *
     * @param[in] key Configuration key to look up
     * @param[in] defaultValue Value returned if key not found or parse fails [default: 0.0]
     *
     * @return Parsed double value, or defaultValue if:
     *         - Key does not exist
     *         - Value string cannot be parsed as double
     *         - Value is NaN/Inf (caught by std::stod exception)
     *
     * @warning Prints warning to std::cerr for parse failures
     * @note Scientific notation supported: 1.23e-4
     */
    double getDouble(const std::string& key, double defaultValue = 0.0) {
        auto it = config_map.find(key);
        if (it != config_map.end()) {
            try {
                return std::stod(it->second);
            } catch (...) {
                std::cerr << "Warning: Invalid double value for key '" << key << "': " << it->second << std::endl;
            }
        }
        return defaultValue;
    }

    /**
     * @brief Retrieve float value for given key
     *
     * Convenience wrapper around getDouble() with float cast. Inherits all
     * parsing behavior from getDouble().
     *
     * @param[in] key Configuration key to look up
     * @param[in] defaultValue Value returned if key not found or parse fails [default: 0.0f]
     *
     * @return Parsed float value (cast from double), or defaultValue
     *
     * @note Precision: float (32-bit) vs double (64-bit) - information loss possible
     * @note Large double values may overflow float range
     */
    float getFloat(const std::string& key, float defaultValue = 0.0f) {
        return static_cast<float>(getDouble(key, defaultValue));
    }

    /**
     * @brief Retrieve integer value for given key
     *
     * Attempts to parse value string as integer using std::stoi. If parsing
     * fails, prints warning and returns default value.
     *
     * @param[in] key Configuration key to look up
     * @param[in] defaultValue Value returned if key not found or parse fails [default: 0]
     *
     * @return Parsed int value, or defaultValue if:
     *         - Key does not exist
     *         - Value string cannot be parsed as integer
     *         - Value contains decimal point (not a valid integer)
     *
     * @warning Prints warning to std::cerr for parse failures
     * @note Fractional values like "10.5" will cause parse error
     * @note Hex/octal notation not supported (use decimal only)
     */
    int getInt(const std::string& key, int defaultValue = 0) {
        auto it = config_map.find(key);
        if (it != config_map.end()) {
            try {
                return std::stoi(it->second);
            } catch (...) {
                std::cerr << "Warning: Invalid int value for key '" << key << "': " << it->second << std::endl;
            }
        }
        return defaultValue;
    }

    /**
     * @brief Retrieve array of strings for given key
     *
     * Parses comma-separated string values. Each substring between commas
     * is trimmed of whitespace and added to result vector.
     *
     * @param[in] key Configuration key to look up
     *
     * @return Vector of strings, one per comma-separated element
     *         - Empty vector if key not found
     *         - Single-element vector if no commas in value
     *
     * @note Empty strings between commas are preserved: "a,,c" → ["a", "", "c"]
     * @note Trailing commas create empty last element: "a,b," → ["a", "b", ""]
     *
     * @example
     * @code
     *   # Config file
     *   files: input1.txt, input2.txt, input3.txt
     *
     *   // Code
     *   auto files = config.getStringArray("files");
     *   // Result: ["input1.txt", "input2.txt", "input3.txt"]
     * @endcode
     */
    std::vector<std::string> getStringArray(const std::string& key) {
        std::vector<std::string> result;
        auto it = config_map.find(key);
        if (it != config_map.end()) {
            std::stringstream ss(it->second);
            std::string item;
            while (std::getline(ss, item, ',')) {
                // Trim whitespace and newline characters
                item.erase(0, item.find_first_not_of(" \t\r\n"));
                item.erase(item.find_last_not_of(" \t\r\n") + 1);
                result.push_back(item);
            }
        }
        return result;
    }

    /**
     * @brief Retrieve array of floats for given key
     *
     * Parses comma-separated numeric values as floats. Each substring is
     * converted using std::stof. Parse failures print warnings and skip
     * that element (not added to result).
     *
     * @param[in] key Configuration key to look up
     *
     * @return Vector of floats, one per successfully parsed element
     *         - Empty vector if key not found or all parses fail
     *         - Partial results if some elements parse successfully
     *
     * @warning Prints warning for each unparseable element
     * @note Scientific notation supported: 1.5e-3, 2.0E+2
     *
     * @example
     * @code
     *   # Config file
     *   coordinates: 37.5, 126.9, 50.0
     *
     *   // Code
     *   auto coords = config.getFloatArray("coordinates");
     *   // Result: [37.5, 126.9, 50.0]
     * @endcode
     */
    std::vector<float> getFloatArray(const std::string& key) {
        std::vector<float> result;
        auto it = config_map.find(key);
        if (it != config_map.end()) {
            std::stringstream ss(it->second);
            std::string item;
            while (std::getline(ss, item, ',')) {
                try {
                    result.push_back(std::stof(item));
                } catch (...) {
                    std::cerr << "Warning: Invalid float value in array for key '" << key << "': " << item << std::endl;
                }
            }
        }
        return result;
    }

    /**
     * @brief Retrieve array of doubles for given key
     *
     * Parses comma-separated numeric values as doubles. Each substring is
     * converted using std::stod. Parse failures print warnings and skip
     * that element (not added to result).
     *
     * @param[in] key Configuration key to look up
     *
     * @return Vector of doubles, one per successfully parsed element
     *         - Empty vector if key not found or all parses fail
     *         - Partial results if some elements parse successfully
     *
     * @warning Prints warning for each unparseable element
     * @note Scientific notation supported: 1.5e-3, 2.0E+2
     * @note Higher precision than getFloatArray() - use for sensitive calculations
     *
     * @example
     * @code
     *   # Config file
     *   decay_constants: 2.42e-6, 1.21e-4, 5.5e-2
     *
     *   // Code
     *   auto lambdas = config.getDoubleArray("decay_constants");
     *   // Result: [2.42e-6, 1.21e-4, 5.5e-2]
     * @endcode
     */
    std::vector<double> getDoubleArray(const std::string& key) {
        std::vector<double> result;
        auto it = config_map.find(key);
        if (it != config_map.end()) {
            std::stringstream ss(it->second);
            std::string item;
            while (std::getline(ss, item, ',')) {
                try {
                    result.push_back(std::stod(item));
                } catch (...) {
                    std::cerr << "Warning: Invalid double value in array for key '" << key << "': " << item << std::endl;
                }
            }
        }
        return result;
    }
};

/**
 * @brief Global configuration reader instance
 *
 * Single global ConfigReader instance shared across the entire application.
 * Provides centralized access to configuration values from any compilation unit.
 *
 * **Usage Pattern:**
 * 1. Call `g_config.loadConfig("file.txt")` once at startup
 * 2. Access values anywhere: `float x = g_config.getFloat("key")`
 *
 * @note Defined as extern here, actual instance defined in ldm.cu
 * @note Not thread-safe during loadConfig() - call before spawning threads
 *
 * @warning Ensure loadConfig() called before any getter access
 * @warning Multiple loadConfig() calls accumulate (keys overwrite)
 */
extern ConfigReader g_config;

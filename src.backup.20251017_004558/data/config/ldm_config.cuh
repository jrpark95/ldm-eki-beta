/**
 * @file ldm_config.cuh
 * @brief Configuration file reader for LDM-EKI simulation system
 *
 * @details Provides ConfigReader class for parsing key-value configuration
 *          files used by the LDM-EKI system. Supports:
 *          - String, int, float, double value types
 *          - Array values (comma-separated)
 *          - Comment lines (starting with #)
 *          - Type-safe value retrieval with defaults
 *
 * @note This is a header-only file - no corresponding .cu implementation
 * @note Uses colon-separated format: KEY: value
 *
 * @example
 *   ConfigReader config;
 *   config.loadConfig("input/setting.txt");
 *   float dt = config.getFloat("time_step", 1.0f);
 *
 * @author LDM-EKI Development Team
 * @date 2025-01-15
 */

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>

class ConfigReader {
private:
    std::map<std::string, std::string> config_map;
    
public:
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
    
    std::string getString(const std::string& key, const std::string& defaultValue = "") {
        auto it = config_map.find(key);
        return (it != config_map.end()) ? it->second : defaultValue;
    }
    
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
    
    float getFloat(const std::string& key, float defaultValue = 0.0f) {
        return static_cast<float>(getDouble(key, defaultValue));
    }
    
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

extern ConfigReader g_config;
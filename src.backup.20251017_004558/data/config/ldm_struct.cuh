/**
 * @file ldm_struct.cuh
 * @brief Core data structures for LDM-EKI simulation system
 *
 * @details This header-only file defines all fundamental data structures
 *          used throughout the LDM-EKI project including:
 *          - Meteorological data structures (PresData, EtasData, UnisData)
 *          - Simulation grid structures (RectangleGrid, Mesh)
 *          - Source and concentration structures
 *          - Type-safe enums for model configuration
 *
 * @note This is a header-only file - no corresponding .cu implementation
 * @note All structures are designed for CPU-side usage
 *
 * @author LDM-EKI Development Team
 * @date 2025-01-15
 */

#pragma once

#include <cmath>
#include <vector>
#include <memory>

// Mathematical constants
constexpr float PI = 3.141592f;
constexpr float PI180 = 0.01745329f;
constexpr float DEG_TO_RAD = PI / 180.0f;
constexpr float RAD_TO_DEG = 180.0f / PI;

// Constants are defined in ldm.cuh as constexpr for CUDA compatibility

// Strong typed enums for better type safety
enum class StabilityCategory : int {
    PASQUILL_GIFFORD = 1,
    BRIGGS_MCELROY_POOLER = 0
};

enum class EnvironmentType : int {
    RURAL = 1,
    URBAN = 0
};

enum class DataSource : int {
    GFS = 1,
    LDAPS = 0
};

struct PresData {
    float DZDT{0.0f};     // No.1 [DZDT] Vertical velocity (m/s)
    float UGRD{0.0f};     // No.2 [UGRD] U-component of wind (m/s)
    float VGRD{0.0f};     // No.3 [VGRD] V-component of wind (m/s)
    float HGT{0.0f};      // No.4 [HGT] Geopotential height (m)
    float TMP{0.0f};      // No.5 [TMP] Temperature (K)
    float RH{0.0f};       // No.7 [RH] Relative Humidity (%)
    float SPFH{0.0f};     // [SPFH] Specific Humidity (kg/kg)
};

struct EtasData {
    float UGRD{0.0f};     // No.1 [UGRD] U-component of wind (m/s)
    float VGRD{0.0f};     // No.2 [VGRD] V-component of wind (m/s)
    float DZDT{0.0f};     // No.6 [DZDT] Vertical velocity (m/s)
    float DEN{0.0f};      // No.7 [DEN] Density of the air (kg/m)
};

struct UnisData {
    float HPBLA{0.0f};    // No.12 [HPBLA] Boundary Layer Depth after B. LAYER (m)
    float T1P5{0.0f};     // No.21 [TMP] Temperature at 1.5m above ground (K)
    float SHTFL{0.0f};    // No.39 [SHTFL] Surface Sensible Heat Flux on Tiles (W/m^2)
    float HTBM{0.0f};     // No.43 [HTBM] Turbulent mixing height after B. Layer (m)
    float HPBL{0.0f};     // No.131 [HPBL] Planetary Boundary Layer Height (m)
    float SFCR{0.0f};     // No.132 [SFCR] Surface Roughness (m)
    float FRICV{0.0f};    // [FRICV] Friction velocity (m/s)
};

struct FlexUnis {
    float HMIX{0.0f};
    float TROP{0.0f};
    float USTR{0.0f};
    float WSTR{0.0f};
    float OBKL{0.0f};
    float LPREC{0.0f};
    float CPREC{0.0f};
    float TCC{0.0f};
    float CLDH{0.0f};
    float VDEP{0.0f};
};

struct FlexPres {
    float DRHO{0.0f};
    float RHO{0.0f};
    float TT{0.0f};
    float UU{0.0f};
    float VV{0.0f};
    float WW{0.0f};
    float QV{0.0f};
    float CLDS{0.0f};
};

struct Source {
    float lat{0.0f};
    float lon{0.0f};
    float height{0.0f};
};

struct Concentration {
    int location{0};
    int sourceterm{0};
    double value{0.0};
};

class RectangleGrid {
private:

public:

    float minX, minY, maxX, maxY;
    float intervalX, intervalY, intervalZ;
    int rows, cols, zdim;

    struct GridPoint{
        float x{0.0f};
        float y{0.0f};
        float z{0.0f};
        float conc{0.0f};
    };

    std::vector<GridPoint> grid;

    RectangleGrid(float _minX, float _minY, float _maxX, float _maxY){

        float width = _maxX - _minX;
        float height = _maxY - _minY;

        minX = _minX - width * 0.5;
        maxX = _maxX + width * 0.5;
        minY = _minY - height * 0.5;
        maxY = _maxY + height * 0.5;

        float grid_size_factor = 3000.0f; // Will be configurable later
        rows = static_cast<int>(std::sqrt(grid_size_factor * (height / width)));
        cols = static_cast<int>(std::sqrt(grid_size_factor * (width / height)));

        intervalX = (maxX - minX) / (cols - 1);
        intervalY = (maxY - minY) / (rows - 1);
        intervalZ = 10.0f; // Will be configurable later

        grid.resize(rows * cols);
        for(int i = 0; i < rows; ++i){
            for(int j = 0; j < cols; ++j){
                int index = i * cols + j;
                grid[index].x = minX + j * intervalX;
                grid[index].y = minY + i * intervalY;
                grid[index].z = 20.0f; // Will be configurable later
                grid[index].conc = 0.0f;
            }
        }

    } 

    ~RectangleGrid() = default;
};

struct MeshPoint {
    float latitude{0.0f};
    float longitude{0.0f};
    float dryDep{0.0f}; // [num_nuclides]
    float wetDep{0.0f}; // [num_nuclides]
};


class Mesh {
public:
    std::vector<std::vector<MeshPoint>> grid;
    int lat_count;
    int lon_count;

    Mesh(float start_lat, float start_lon,
        float lat_step, float lon_step,
         int lat_num, int lon_num)
        : lat_count(lat_num),
          lon_count(lon_num)
    {
        grid.resize(lat_count, std::vector<MeshPoint>(lon_count));
        
        for (int i = 0; i < lat_count; ++i) {
            for (int j = 0; j < lon_count; ++j) {
                grid[i][j].latitude = start_lat + i * lat_step;
                grid[i][j].longitude = start_lon + j * lon_step;

                grid[i][j].dryDep = 0.0f;
                grid[i][j].wetDep = 0.0f;
            }
        }
    }
    
};

/******************************************************************************
 * @file ldm_struct.cuh
 * @brief Core data structures for LDM-EKI simulation system
 *
 * @details This header-only file defines all fundamental data structures
 *          used throughout the LDM-EKI project. It provides:
 *
 *          **Meteorological Data Structures:**
 *          - PresData: Pressure-level atmospheric variables (3D grid)
 *          - EtasData: Eta-level atmospheric variables
 *          - UnisData: Uniform (2D surface) meteorological fields
 *          - FlexPres/FlexUnis: FLEXPART-format meteorological data
 *
 *          **Simulation Domain Structures:**
 *          - RectangleGrid: Adaptive rectangular grid for concentration output
 *          - Mesh: Regular lat/lon mesh for deposition calculations
 *
 *          **Source and Observation Structures:**
 *          - Source: Emission source location definition
 *          - Concentration: Receptor observation data
 *
 *          **Type-Safe Enumerations:**
 *          - StabilityCategory: Atmospheric stability schemes
 *          - EnvironmentType: Rural vs urban surface roughness
 *          - DataSource: Meteorological data provider (GFS/LDAPS)
 *
 * @note This is a header-only file - no corresponding .cu implementation
 * @note All structures are designed for CPU-side usage
 * @note GPU kernels receive these structures through kernel parameter passing
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/

#pragma once

#include <cmath>
#include <vector>
#include <memory>

// ===========================================================================
// MATHEMATICAL CONSTANTS
// ===========================================================================

constexpr float PI = 3.141592f;           ///< Pi constant
constexpr float PI180 = 0.01745329f;      ///< Degrees to radians factor (PI/180)
constexpr float DEG_TO_RAD = PI / 180.0f; ///< Degrees to radians conversion
constexpr float RAD_TO_DEG = 180.0f / PI; ///< Radians to degrees conversion

// ===========================================================================
// TYPE-SAFE ENUMERATIONS
// ===========================================================================

/**
 * @brief Atmospheric stability categorization schemes
 *
 * Determines which empirical formulas are used to estimate turbulent
 * dispersion coefficients. Different schemes are appropriate for different
 * meteorological conditions and terrain types.
 */
enum class StabilityCategory : int {
    PASQUILL_GIFFORD = 1,      ///< Pasquill-Gifford stability classes (A-F)
    BRIGGS_MCELROY_POOLER = 0  ///< Briggs-McElroy-Pooler scheme
};

/**
 * @brief Surface environment type classification
 *
 * Affects surface roughness length and turbulence parameterizations.
 * Urban areas have higher roughness (buildings, obstacles) leading to
 * enhanced mechanical turbulence.
 */
enum class EnvironmentType : int {
    RURAL = 1,  ///< Rural terrain (fields, forests, low roughness)
    URBAN = 0   ///< Urban terrain (buildings, high roughness)
};

/**
 * @brief Meteorological data source provider
 *
 * Specifies which numerical weather prediction (NWP) model data is being
 * used. Different models have different grid structures, vertical coordinates,
 * and file formats.
 */
enum class DataSource : int {
    GFS = 1,    ///< NCEP Global Forecast System (global, 0.5° resolution)
    LDAPS = 0   ///< KMA Local Data Assimilation Prediction System (regional)
};

// ===========================================================================
// METEOROLOGICAL DATA STRUCTURES
// ===========================================================================

/**
 * @brief Pressure-level meteorological data
 *
 * Contains atmospheric variables defined on isobaric (constant pressure)
 * surfaces. This is the standard vertical coordinate system used by most
 * global NWP models like GFS.
 *
 * @note Data layout: 3D array [lon x lat x pressure_level]
 * @note All fields are initialized to zero for safety
 */
struct PresData {
    float DZDT{0.0f};   ///< Vertical velocity [Pa/s] - Positive: upward motion
    float UGRD{0.0f};   ///< Zonal (eastward) wind component [m/s]
    float VGRD{0.0f};   ///< Meridional (northward) wind component [m/s]
    float HGT{0.0f};    ///< Geopotential height [m] - Height of pressure surface above sea level
    float TMP{0.0f};    ///< Temperature [K] - Absolute temperature
    float RH{0.0f};     ///< Relative humidity [%] - Range: 0-100
    float SPFH{0.0f};   ///< Specific humidity [kg/kg] - Mass mixing ratio of water vapor
};

/**
 * @brief Eta-level meteorological data
 *
 * Contains atmospheric variables defined on terrain-following hybrid
 * coordinates (eta levels). Commonly used by regional models like LDAPS
 * to better represent complex terrain.
 *
 * @note Eta coordinate: σ = (p - p_top) / (p_surface - p_top)
 * @note Better surface layer resolution than pressure levels
 */
struct EtasData {
    float UGRD{0.0f};   ///< Zonal wind component [m/s]
    float VGRD{0.0f};   ///< Meridional wind component [m/s]
    float DZDT{0.0f};   ///< Vertical velocity [m/s] - Terrain-following coordinate
    float DEN{0.0f};    ///< Air density [kg/m³] - Derived from pressure and temperature
};

/**
 * @brief Surface (uniform-level) meteorological data
 *
 * Contains 2D fields defined at the surface or averaged over the boundary
 * layer. These variables are critical for turbulence parameterization and
 * surface exchange processes.
 *
 * @note All fields are 2D arrays [lon x lat]
 * @note Boundary layer parameters updated every 3-6 hours in typical NWP output
 */
struct UnisData {
    float HPBLA{0.0f};  ///< Boundary layer depth after B.LAYER scheme [m] - Diagnostic value
    float T1P5{0.0f};   ///< Temperature at 1.5m above ground [K] - Screen-level temperature
    float SHTFL{0.0f};  ///< Surface sensible heat flux [W/m²] - Positive: upward heat transfer
    float HTBM{0.0f};   ///< Turbulent mixing height from B.LAYER [m] - Active mixing depth
    float HPBL{0.0f};   ///< Planetary boundary layer height [m] - Top of well-mixed layer
    float SFCR{0.0f};   ///< Surface roughness length [m] - Logarithmic profile parameter
    float FRICV{0.0f};  ///< Friction velocity u* [m/s] - Momentum flux scale
};

/**
 * @brief FLEXPART-format surface meteorological data
 *
 * Meteorological fields in the format expected by FLEXPART (FLEXible
 * PARTicle dispersion model). This structure matches FLEXPART's internal
 * data representation for efficient interoperability.
 *
 * @note Used for FLEXPART-compatible output mode
 * @note 2D fields defined at surface level
 */
struct FlexUnis {
    float HMIX{0.0f};   ///< Mixing height [m] - Well-mixed boundary layer depth
    float TROP{0.0f};   ///< Tropopause height [m] - Thermal tropopause altitude
    float USTR{0.0f};   ///< Friction velocity u* [m/s] - Surface momentum flux scale
    float WSTR{0.0f};   ///< Convective velocity scale w* [m/s] - Buoyancy-driven turbulence
    float OBKL{0.0f};   ///< Monin-Obukhov length L [m] - Stability parameter (L<0: unstable, L>0: stable)
    float LPREC{0.0f};  ///< Large-scale precipitation rate [m/s] - Stratiform rain/snow
    float CPREC{0.0f};  ///< Convective precipitation rate [m/s] - Shower/thunderstorm
    float TCC{0.0f};    ///< Total cloud cover [0-1] - Fraction of sky covered by clouds
    float CLDH{0.0f};   ///< Cloud top height [m] - Highest cloud boundary
    float VDEP{0.0f};   ///< Dry deposition velocity [m/s] - Surface removal rate
};

/**
 * @brief FLEXPART-format pressure-level meteorological data
 *
 * 3D atmospheric fields in FLEXPART-compatible format. Contains wind,
 * thermodynamic, and microphysical variables needed for particle transport
 * and wet deposition calculations.
 *
 * @note 3D fields: [lon x lat x level]
 * @note Vertical coordinate: Pressure levels or height levels depending on model
 */
struct FlexPres {
    float DRHO{0.0f};   ///< Vertical density gradient [kg/m⁴] - dρ/dz for vertical advection
    float RHO{0.0f};    ///< Air density [kg/m³] - From equation of state
    float TT{0.0f};     ///< Temperature [K] - Absolute temperature
    float UU{0.0f};     ///< Zonal wind component [m/s] - Eastward
    float VV{0.0f};     ///< Meridional wind component [m/s] - Northward
    float WW{0.0f};     ///< Vertical velocity [m/s] - Upward positive
    float QV{0.0f};     ///< Water vapor mixing ratio [kg/kg] - Mass of vapor per mass of dry air
    float CLDS{0.0f};   ///< Cloud water content [kg/m³] - Liquid + ice cloud condensate
};

// ===========================================================================
// SIMULATION DOMAIN STRUCTURES
// ===========================================================================

/**
 * @brief Emission source location specification
 *
 * Defines the geographic location and height of a radioactive emission source.
 * Multiple sources can be specified for complex release scenarios.
 *
 * @note Coordinates use geographic (lat/lon) system
 * @note Height is relative to mean sea level (MSL)
 */
struct Source {
    float lat{0.0f};    ///< Latitude [degrees] - Range: [-90, 90], positive north
    float lon{0.0f};    ///< Longitude [degrees] - Range: [-180, 180], positive east
    float height{0.0f}; ///< Release height [m MSL] - Altitude above sea level
};

/**
 * @brief Receptor concentration observation
 *
 * Stores a concentration measurement at a specific receptor location and
 * time. Used for data assimilation and source inversion.
 *
 * @note Part of ensemble Kalman filter observation vector
 */
struct Concentration {
    int location{0};     ///< Receptor index - Corresponds to receptor array position
    int sourceterm{0};   ///< Source term index - Which emission scenario
    double value{0.0};   ///< Observed concentration [Bq/m³] or dose rate [Sv/h]
};

/**
 * @brief Adaptive rectangular grid for concentration output
 *
 * Implements an automatically-sized rectangular grid that extends beyond
 * the simulation domain to capture long-range transport. Grid resolution
 * is dynamically calculated based on domain aspect ratio.
 *
 * @note Grid automatically expands by 50% in each direction beyond min/max bounds
 * @note Resolution determined by grid_size_factor (default: 3000 cells)
 * @note Used for VTK visualization output
 */
class RectangleGrid {
private:
    // No private members currently - future extension point for cached calculations

public:
    float minX;         ///< Minimum longitude [degrees] - Western boundary (expanded)
    float minY;         ///< Minimum latitude [degrees] - Southern boundary (expanded)
    float maxX;         ///< Maximum longitude [degrees] - Eastern boundary (expanded)
    float maxY;         ///< Maximum latitude [degrees] - Northern boundary (expanded)

    float intervalX;    ///< Longitude grid spacing [degrees] - Uniform horizontal resolution
    float intervalY;    ///< Latitude grid spacing [degrees] - Uniform vertical resolution
    float intervalZ;    ///< Vertical grid spacing [m] - Currently fixed at 10m

    int rows;           ///< Number of latitude grid points - Determined by aspect ratio
    int cols;           ///< Number of longitude grid points - Determined by aspect ratio
    int zdim;           ///< Number of vertical levels - Not currently used in output

    /**
     * @brief Individual grid point with accumulated concentration
     *
     * Each grid cell stores its center coordinates and the cumulative
     * concentration deposited by particles during simulation.
     */
    struct GridPoint {
        float x{0.0f};      ///< Grid point longitude [degrees]
        float y{0.0f};      ///< Grid point latitude [degrees]
        float z{0.0f};      ///< Grid point height [m MSL]
        float conc{0.0f};   ///< Accumulated concentration [Bq/m³] or time-integrated [Bq·s/m³]
    };

    std::vector<GridPoint> grid;  ///< Flattened grid array [rows * cols] - Row-major order

    /**
     * @brief Construct rectangular grid with automatic sizing
     *
     * Creates a grid that extends 50% beyond the provided bounds in all
     * directions. Grid resolution is calculated to maintain approximately
     * 3000 total grid cells while respecting domain aspect ratio.
     *
     * @param[in] _minX Minimum longitude of core domain [degrees]
     * @param[in] _minY Minimum latitude of core domain [degrees]
     * @param[in] _maxX Maximum longitude of core domain [degrees]
     * @param[in] _maxY Maximum latitude of core domain [degrees]
     *
     * @note Grid size formula: N = sqrt(3000 * aspect_ratio)
     * @note Vertical spacing (intervalZ) currently fixed at 10m
     * @note All grid points initialized with z=20m and conc=0
     */
    RectangleGrid(float _minX, float _minY, float _maxX, float _maxY) {
        float width = _maxX - _minX;
        float height = _maxY - _minY;

        // Expand domain by 50% in each direction
        minX = _minX - width * 0.5;
        maxX = _maxX + width * 0.5;
        minY = _minY - height * 0.5;
        maxY = _maxY + height * 0.5;

        // Calculate grid dimensions maintaining aspect ratio
        float grid_size_factor = 3000.0f;
        rows = static_cast<int>(std::sqrt(grid_size_factor * (height / width)));
        cols = static_cast<int>(std::sqrt(grid_size_factor * (width / height)));

        // Compute uniform grid spacing
        intervalX = (maxX - minX) / (cols - 1);
        intervalY = (maxY - minY) / (rows - 1);
        intervalZ = 10.0f;

        // Initialize grid points
        grid.resize(rows * cols);
        for(int i = 0; i < rows; ++i) {
            for(int j = 0; j < cols; ++j) {
                int index = i * cols + j;
                grid[index].x = minX + j * intervalX;
                grid[index].y = minY + i * intervalY;
                grid[index].z = 20.0f;  // Fixed output height
                grid[index].conc = 0.0f;
            }
        }
    }

    ~RectangleGrid() = default;
};

/**
 * @brief Individual mesh point for deposition calculations
 *
 * Stores geographic location and cumulative deposition amounts for both
 * dry and wet deposition processes. Used in regular lat/lon mesh arrays.
 */
struct MeshPoint {
    float latitude{0.0f};   ///< Latitude [degrees] - Range: [-90, 90]
    float longitude{0.0f};  ///< Longitude [degrees] - Range: [-180, 180]
    float dryDep{0.0f};     ///< Dry deposition [Bq/m²] - Surface deposition by gravitational settling
    float wetDep{0.0f};     ///< Wet deposition [Bq/m²] - Scavenging by precipitation
};

/**
 * @brief Regular lat/lon mesh for deposition output
 *
 * Implements a uniform geographic grid for tracking dry and wet deposition.
 * Unlike RectangleGrid, this uses fixed spacing without automatic sizing.
 * Commonly used for generating deposition maps in post-processing.
 *
 * @note Grid spacing and bounds provided at construction
 * @note All deposition values initialized to zero
 */
class Mesh {
public:
    std::vector<std::vector<MeshPoint>> grid;  ///< 2D mesh array [lat][lon] - Natural indexing
    int lat_count;  ///< Number of latitude points - Y dimension
    int lon_count;  ///< Number of longitude points - X dimension

    /**
     * @brief Construct uniform lat/lon deposition mesh
     *
     * Creates a regular geographic grid with specified spacing and extent.
     * Each mesh point is initialized with its coordinates and zero deposition.
     *
     * @param[in] start_lat Starting latitude [degrees] - Southwestern corner
     * @param[in] start_lon Starting longitude [degrees] - Southwestern corner
     * @param[in] lat_step Latitude spacing [degrees] - Positive northward
     * @param[in] lon_step Longitude spacing [degrees] - Positive eastward
     * @param[in] lat_num Number of latitude points - Y dimension size
     * @param[in] lon_num Number of longitude points - X dimension size
     *
     * @note Grid covers: [start_lat, start_lat + (lat_num-1)*lat_step] x
     *                    [start_lon, start_lon + (lon_num-1)*lon_step]
     * @note All deposition fields initialized to 0.0
     */
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

////////////////////////////////////////////////////////////////////////////////
/// @file params.hpp
/// @brief Kernel parameter structure for GPU computations
/// @author Juryong Park
/// @date 2025
///
/// Defines the KernelScalars structure that aggregates all scalar parameters
/// needed by CUDA kernels. This structure replaced global __constant__ memory
/// variables to achieve compatibility with non-relocatable device code (non-RDC)
/// compilation mode.
///
/// @note The structure is aligned to 16 bytes for optimal GPU memory access.
///       Field names are chosen to avoid macro conflicts with legacy code.
////////////////////////////////////////////////////////////////////////////////

#pragma once

/// @brief Kernel scalar parameters passed to all GPU kernels
///
/// This structure replaces __constant__ scalar variables to avoid "invalid device
/// symbol" errors in non-RDC compilation mode. All simulation switches, physical
/// constants, and data pointers are bundled here for efficient kernel launches.
///
/// @note Alignment ensures coalesced memory access on GPU
/// @note Field names avoid preprocessor macro conflicts (e.g., nop, isRural, isPG)
struct alignas(16) KernelScalars {
    // -------------------------------------------------------------------------
    // Simulation Switches
    // -------------------------------------------------------------------------
    int turb_switch;    ///< Enable turbulence model (0=off, 1=on)
    int drydep;         ///< Enable dry deposition (0=off, 1=on)
    int wetdep;         ///< Enable wet deposition (0=off, 1=on)
    int raddecay;       ///< Enable radioactive decay (0=off, 1=on)

    // -------------------------------------------------------------------------
    // Particle and Model Parameters
    // -------------------------------------------------------------------------
    int num_particles;  ///< Total number of particles (legacy: nop)
    int is_rural;       ///< Rural dispersion category flag (legacy: isRural)
    int is_pg;          ///< Pasquill-Gifford stability class flag (legacy: isPG)
    int is_gfs;         ///< GFS meteorological data flag (legacy: isGFS)

    // -------------------------------------------------------------------------
    // Time and Grid Parameters
    // -------------------------------------------------------------------------
    float delta_time;     ///< Time step size in seconds (legacy: dt)
    float grid_start_lat; ///< Grid origin latitude in degrees (legacy: start_lat)
    float grid_start_lon; ///< Grid origin longitude in degrees (legacy: start_lon)
    float grid_lat_step;  ///< Latitude resolution in degrees (legacy: lat_step)
    float grid_lon_step;  ///< Longitude resolution in degrees (legacy: lon_step)

    // -------------------------------------------------------------------------
    // Physics Parameters
    // -------------------------------------------------------------------------
    float settling_vel;   ///< Gravitational settling velocity in m/s (legacy: vsetaver)
    float cunningham_fac; ///< Cunningham slip correction factor (legacy: cunningham)

    // -------------------------------------------------------------------------
    // Device Memory Pointers
    // -------------------------------------------------------------------------
    const float* T_matrix; ///< CRAM decay transition matrix (N_NUCLIDES × N_NUCLIDES)
    const float* height_levels; ///< Vertical height levels array (dimZ_GFS elements)
};

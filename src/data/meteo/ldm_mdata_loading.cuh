/******************************************************************************
 * @file ldm_mdata_loading.cuh
 * @brief Meteorological data loading module header
 *
 * @details Forward declaration header for meteorological data loading functions.
 *          This is a minimal header that only declares the LDM class to avoid
 *          circular dependencies. Actual method declarations are in ldm.cuh.
 *
 *          **Module Functions (declared in ldm.cuh):**
 *          - LDM::initializeFlexGFSData() - Load first three timesteps
 *          - LDM::loadFlexGFSData() - Load next timestep with rolling buffer
 *          - LDM::loadFlexHeightData() - Load initial vertical grid
 *
 *          **Implementation:**
 *          - See ldm_mdata_loading.cu for full implementations
 *
 * @note Header-only file with forward declaration
 * @note Part of modularized meteorological data management system
 * @note Include ldm.cuh for actual method declarations
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/

#pragma once

// Forward declaration only - method declarations are in ldm.cuh
class LDM;
